import argparse
from . import convert_coco_labels, copy_dataset, convert_coco_dataset

def main():
    """Command line interface entry point"""
    parser = argparse.ArgumentParser(description='Convert COCO dataset to YOLO format and handle Kaggle storage limitations')
    
    # Common parameters
    parser.add_argument('--json-dir', type=str, 
                        default='/kaggle/input/coco-2017-dataset/coco2017/annotations',
                        help='COCO JSON annotation directory')
    
    parser.add_argument('--output-dir', type=str, 
                        default='/kaggle/working/coco_yolo',
                        help='Output directory for YOLO labels')
    
    # Optional parameters
    parser.add_argument('--final-dest', type=str, 
                        default='/kaggle/tmp/COCO2017',
                        help='Final dataset directory (when copying files)')
    
    parser.add_argument('--use-segments', action='store_true',
                        help='Whether to use segment annotations')
    
    parser.add_argument('--no-cls91to80', action='store_false', dest='cls91to80',
                        help='Disable mapping 91 classes to 80 classes (enabled by default)')
    
    parser.add_argument('--max-workers', type=int, default=8,
                        help='Number of parallel worker threads')
    
    # Directory structure customization
    parser.add_argument('--train-dir-name', type=str, default='train2017',
                        help='Destination train subdirectory name')
    
    parser.add_argument('--val-dir-name', type=str, default='val2017',
                        help='Destination validation subdirectory name')
    
    parser.add_argument('--src-train-dir', type=str, default='train2017',
                        help='Source train directory name')
    
    parser.add_argument('--src-val-dir', type=str, default='val2017',
                        help='Source validation directory name')
    
    # Command mode selection
    parser.add_argument('--mode', type=str, choices=['convert', 'copy', 'all'], default='convert',
                        help='Operation mode: convert (labels only), copy (after convert), or all (both)')
    
    # Set cls91to80 to True by default
    parser.set_defaults(cls91to80=True)
    
    args = parser.parse_args()
    
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

if __name__ == "__main__":
    main()