def run_pipeline(target_folder_name):
    root_dir = os.path.join(os.getcwd(), "source")
    data_loader = DataLoader(root_dir, target_folder_name)
    
    print(f"Processing folder: {data_loader.sample_folder}")

    valid_sets, errors = data_loader.get_file_pairs(data_loader.sample_folder)

    if not valid_sets:
        print("No valid data found.")
        if errors:
            print("Errors encountered:")
            for e in errors: print(f" - {e}")
        return

    print(f"Found {len(valid_sets)} valid image sets to process.")
    
    # Create Output Directory
    output_dir = os.path.join(data_loader.sample_folder, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the valid sets
    for item in valid_sets:
        sample_id = item['id']
        bf_path = item['bf']
        fl_path = item['fl']
        xml_path = item['xml']
        
        print(f"--> Processing ID: {sample_id}")
        
        # 1. Parse Metadata
        pixel_size_x, pixel_size_y, unit, meta_bit_depth = parse_metadata(xml_path)
        
        if pixel_size_x and pixel_size_y:
            pixel_size = (pixel_size_x + pixel_size_y) / 2.0
            area_factor = pixel_size ** 2
            unit_str = unit if unit else 'um'
            has_calibration = True
        else:
            pixel_size = None
            area_factor = 1.0
            unit_str = 'pixels'
            has_calibration = False

        # 2. Load Images (WITH ERROR CHECKING)
        img_bf = cv2.imread(bf_path, cv2.IMREAD_UNCHANGED)
        if img_bf is None:
            print(f"  ⚠ Error: Could not load brightfield image: {bf_path}")
            continue # Skip this file
            
        if img_bf.ndim == 3: img_bf = cv2.cvtColor(img_bf, cv2.COLOR_BGR2GRAY)
        
        img_red_orig = cv2.imread(fl_path, cv2.IMREAD_UNCHANGED)
        if img_red_orig is None:
            print(f"  ⚠ Error: Could not load fluorescence image: {fl_path}")
            continue # Skip this file

        if img_red_orig.ndim == 3: img_red_orig = cv2.cvtColor(img_red_orig, cv2.COLOR_BGR2GRAY)

        # 3. Handle Bit Depth
        orig_dtype = img_red_orig.dtype
        orig_max = img_red_orig.max()
        
        if orig_dtype == np.uint16:
            if orig_max <= 4095: bit_depth = 12; max_val = 4095
            elif orig_max <= 16383: bit_depth = 14; max_val = 16383
            else: bit_depth = 16; max_val = 65535
            bit_conv_factor = max_val / 255.0
            
            # Create 8-bit versions for segmentation/vis
            img_bf_8bit = np.zeros_like(img_bf, dtype=np.uint8)
            cv2.normalize(img_bf, img_bf_8bit, 0, 255, cv2.NORM_MINMAX)
            img_bf_proc = img_bf_8bit
            
            img_red_8bit = np.zeros_like(img_red_orig, dtype=np.uint8)
            cv2.normalize(img_red_orig, img_red_8bit, 0, 255, cv2.NORM_MINMAX)
        else:
            bit_depth = 8; max_val = 255; bit_conv_factor = 1.0
            img_bf_proc = img_bf
            img_red_8bit = img_red_orig.copy()

        # 4. Enhance Red for Visualization
        img_red_enh = adjust_red_channel(img_red_8bit, RED_NORMALIZE, RED_BRIGHTNESS, RED_GAMMA)

        # 5. Segmentation (Brightfield)
        bg = cv2.GaussianBlur(img_bf_proc, (0, 0), sigmaX=GAUSSIAN_SIGMA, sigmaY=GAUSSIAN_SIGMA)
        enhanced = cv2.subtract(bg, img_bf_proc)
        enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        _, thresh = cv2.threshold(enhanced_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
        closed = cv2.dilate(closed, kernel, iterations=DILATE_ITERATIONS)
        closed = cv2.erode(closed, kernel, iterations=ERODE_ITERATIONS)
        
        # Find Contours
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        solid = np.where(labels > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter Contours
        filtered = [c for c in contours if MIN_OBJECT_AREA <= cv2.contourArea(c) <= MAX_OBJECT_AREA]
        
        # 6. Measurements
        object_data = []
        for c in filtered:
            area_px = cv2.contourArea(c)
            perimeter_px = cv2.arcLength(c, True)
            
            area_phys = area_px * area_factor if has_calibration else area_px
            perim_phys = perimeter_px * pixel_size if has_calibration else perimeter_px
            
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            
            mask = np.zeros_like(img_red_8bit, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # Measure on ORIGINAL high-bit image
            red_px_orig = img_red_orig[mask == 255].astype(np.float64)
            if len(red_px_orig) == 0: continue
            
            total_int = float(np.sum(red_px_orig))
            mean_int = float(np.mean(red_px_orig))
            std_int = float(np.std(red_px_orig))
            
            if total_int == 0: continue
            
            int_per_area = total_int / area_phys
            
            object_data.append({
                'object_id': 0, # Placeholder
                'contour': c,
                'centroid_x': cx, 'centroid_y': cy,
                'area_phys': area_phys, 'perim_phys': perim_phys,
                'total_int': total_int, 'mean_int': mean_int, 'std_int': std_int,
                'intensity_per_area_orig': int_per_area
            })

        # Sort and ID
        object_data.sort(key=lambda x: x['intensity_per_area_orig'], reverse=True)
        for i, obj in enumerate(object_data, 1): obj['object_id'] = i
        
        if not object_data:
            print(f"  No valid objects found for {sample_id}")
            continue

        # 7. Generate Outputs (Excel & Plots)
        excel_name = f"{sample_id}_statistics.xlsx"
        excel_path = os.path.join(output_dir, excel_name)
        wb = Workbook()
        ws = wb.active
        ws.title = "Data"
        
        headers = ['ID', f'Area ({unit_str}²)', f'Total Int ({bit_depth}-bit)', f'Mean Int ({bit_depth}-bit)', f'Int/{unit_str}²']
        ws.append(headers)
        style_header(ws, 1)
        
        for obj in object_data:
            ws.append([
                obj['object_id'], round(obj['area_phys'], 2), 
                round(obj['total_int'], 1), round(obj['mean_int'], 2), 
                round(obj['intensity_per_area_orig'], 2)
            ])
            
        # Generate Plots
        plot_err = create_error_bar_plot(object_data, unit_str, ERROR_PERCENTAGE)
        plot_bar = create_bar_chart_with_errors(object_data, unit_str, ERROR_PERCENTAGE)
        plot_stat = create_statistics_plot(object_data, unit_str)
        
        if plot_err:
            img = XLImage(plot_err)
            ws.add_image(img, 'G2')
        if plot_bar:
            img = XLImage(plot_bar)
            ws.add_image(img, 'G25')
            
        auto_adjust_column_width(ws)
        wb.save(excel_path)
        
        # 8. Save Debug Images (Overlay)
        vis_overlay = cv2.cvtColor(img_bf_proc, cv2.COLOR_GRAY2BGR)
        red_overlay = np.zeros_like(vis_overlay)
        red_overlay[:, :, 2] = img_red_8bit # Use 8-bit for display
        vis_overlay = cv2.addWeighted(vis_overlay, 0.7, red_overlay, 0.3, 0)
        
        for obj in object_data:
            cv2.drawContours(vis_overlay, [obj['contour']], -1, (0, 255, 0), 1)
            cv2.putText(vis_overlay, str(obj['object_id']), (obj['centroid_x'], obj['centroid_y']),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        save_debug_image(output_dir, f"{sample_id}_overlay.png", vis_overlay, pixel_size, unit_str)
        print(f"  ✓ Saved results to {output_dir}")

    print("Pipeline completed.")