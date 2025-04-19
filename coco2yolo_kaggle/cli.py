import argparse
from . import convert_coco_labels, copy_dataset, convert_coco_dataset
from . import convert_yolo_labels, convert_yolo_dataset

def main():
    """Command line interface entry point"""
    parser = argparse.ArgumentParser(description='Convert between COCO and YOLO formats and handle Kaggle storage limitations')
    
    # Add parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--output-dir', type=str, 
                        default='/kaggle/working',
                        help='Output directory')
    parent_parser.add_argument('--final-dest', type=str, 
                        default='/kaggle/tmp/dataset',
                        help='Final dataset directory (when copying files)')
    parent_parser.add_argument('--max-workers', type=int, default=8,
                        help='Number of parallel worker threads')
    
    # Create subparsers for different conversion directions
    subparsers = parser.add_subparsers(dest='direction', help='Conversion direction')
    
    # COCO to YOLO subparser
    coco2yolo_parser = subparsers.add_parser('coco2yolo', help='Convert COCO format to YOLO format',
                                           parents=[parent_parser])
    
    # COCO to YOLO specific arguments
    coco2yolo_parser.add_argument('--json-dir', type=str, 
                        default='/kaggle/input/coco-2017-dataset/coco2017/annotations',
                        help='COCO JSON annotation directory')
    
    coco2yolo_parser.add_argument('--use-segments', action='store_true',
                        help='Whether to use segment annotations')
    
    coco2yolo_parser.add_argument('--no-cls91to80', action='store_false', dest='cls91to80',
                        help='Disable mapping 91 classes to 80 classes (enabled by default)')
    
    # Directory structure customization for COCO to YOLO
    coco2yolo_parser.add_argument('--train-dir-name', type=str, default='train2017',
                        help='Destination train subdirectory name')
    
    coco2yolo_parser.add_argument('--val-dir-name', type=str, default='val2017',
                        help='Destination validation subdirectory name')
    
    coco2yolo_parser.add_argument('--src-train-dir', type=str, default='train2017',
                        help='Source train directory name')
    
    coco2yolo_parser.add_argument('--src-val-dir', type=str, default='val2017',
                        help='Source validation directory name')
    
    # Command mode selection for COCO to YOLO
    coco2yolo_parser.add_argument('--mode', type=str, choices=['convert', 'copy', 'all'], default='convert',
                        help='Operation mode: convert (labels only), copy (after convert), or all (both)')
    
    # Set cls91to80 to True by default for COCO to YOLO
    coco2yolo_parser.set_defaults(cls91to80=True)
    
    # YOLO to COCO subparser
    yolo2coco_parser = subparsers.add_parser('yolo2coco', help='Convert YOLO format to COCO format',
                                           parents=[parent_parser])
    
    # YOLO to COCO specific arguments
    yolo2coco_parser.add_argument('--yolo-dataset-path', type=str, 
                        default='/kaggle/input/yolo-dataset',
                        help='YOLO dataset path with labels/{train,val} and images/{train,val} structure')
    
    yolo2coco_parser.add_argument('--yaml-path', type=str, 
                        help='Path to YAML file with class names')
    
    yolo2coco_parser.add_argument('--class-names', type=str, 
                        help='Comma-separated list of class names (alternative to --yaml-path)')
    
    yolo2coco_parser.add_argument('--split-names', type=str, default='train,val',
                        help='Comma-separated list of dataset splits to process')
    
    yolo2coco_parser.add_argument('--image-ext', type=str, default='jpg',
                        help='Image file extension')
    
    yolo2coco_parser.add_argument('--copy-images', action='store_true',
                        help='Copy images to final destination')
    
    args = parser.parse_args()
    
    # If no direction specified, show help
    if args.direction is None:
        parser.print_help()
        return
    
    # Handle COCO to YOLO conversions
    if args.direction == 'coco2yolo':
        if args.mode == 'convert':
            # Only convert labels
            convert_coco_labels(
                json_dir=args.json_dir,
                output_dir=args.output_dir,
                use_segments=args.use_segments,
                cls91to80=args.cls91to80
            )
        elif args.mode == 'copy':
            # Only copy files (assumes labels are already converted)
            copy_dataset(
                json_dir=args.json_dir,
                output_dir=args.output_dir,
                final_dest=args.final_dest,
                max_workers=args.max_workers,
                train_dir_name=args.train_dir_name,
                val_dir_name=args.val_dir_name,
                src_train_dir_name=args.src_train_dir,
                src_val_dir_name=args.src_val_dir
            )
        elif args.mode == 'all':
            # Do both conversion and copying
            convert_coco_dataset(
                json_dir=args.json_dir,
                output_dir=args.output_dir,
                final_dest=args.final_dest,
                use_segments=args.use_segments,
                cls91to80=args.cls91to80,
                max_workers=args.max_workers,
                copy_files=True,
                train_dir_name=args.train_dir_name,
                val_dir_name=args.val_dir_name,
                src_train_dir_name=args.src_train_dir,
                src_val_dir_name=args.src_val_dir
            )
    
    # Handle YOLO to COCO conversions
    elif args.direction == 'yolo2coco':
        # Process class names if provided as a comma-separated string
        class_names = None
        if args.class_names:
            class_names = [name.strip() for name in args.class_names.split(',')]
        
        # Process split names
        split_names = [name.strip() for name in args.split_names.split(',')]
        
        if args.copy_images:
            # Full dataset conversion including copying images
            convert_yolo_dataset(
                yolo_dataset_path=args.yolo_dataset_path,
                output_dir=args.output_dir,
                final_dest=args.final_dest,
                class_names=class_names,
                yaml_path=args.yaml_path,
                split_names=split_names,
                image_ext=args.image_ext,
                copy_images=True,
                max_workers=args.max_workers
            )
        else:
            # Labels conversion only
            convert_yolo_labels(
                yolo_dataset_path=args.yolo_dataset_path,
                output_dir=args.output_dir,
                class_names=class_names,
                yaml_path=args.yaml_path,
                split_names=split_names,
                image_ext=args.image_ext
            )

if __name__ == "__main__":
    main()