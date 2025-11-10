#!/usr/bin/env python3
"""
Downscale images from 512x512 to 256x256
"""

import argparse
from pathlib import Path
from PIL import Image
import sys


def downscale_image(input_path, output_path=None, size=(256, 256), resampling=Image.LANCZOS):
    """
    Downscale an image to the specified size.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image (optional)
        size: Target size as (width, height) tuple
        resampling: Resampling filter (LANCZOS, BILINEAR, BICUBIC, etc.)
    
    Returns:
        Path to output image
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    # Load image
    img = Image.open(input_path)
    
    # Check if image is already 512x512 (optional warning)
    if img.size != (512, 512):
        print(f"Warning: Input image size is {img.size}, expected (512, 512)")
    
    # Downscale image
    img_downscaled = img.resize(size, resampling)
    
    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_256{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Save downscaled image
    img_downscaled.save(output_path)
    print(f"Saved: {output_path}")
    
    return output_path


def batch_downscale(input_dir, output_dir=None, pattern="*.png", size=(256, 256)):
    """
    Batch downscale all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images (optional)
        pattern: File pattern to match (e.g., "*.png", "*.jpg")
        size: Target size as (width, height) tuple
    """
    input_dir = Path(input_dir)
    
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    
    # Create output directory if specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching images
    image_files = list(input_dir.glob(pattern))
    
    if not image_files:
        print(f"No images found matching pattern: {pattern}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        try:
            if output_dir is not None:
                out_path = output_dir / img_path.name
            else:
                out_path = None
            
            downscale_image(img_path, out_path, size)
            print(f"Progress: {i}/{len(image_files)}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}", file=sys.stderr)
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Downscale images from 512x512 to 256x256",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python downscale_image.py input.png
  python downscale_image.py input.png -o output.png
  
  # Batch process directory
  python downscale_image.py input_dir/ --batch --pattern "*.png"
  python downscale_image.py input_dir/ --batch --output-dir output_dir/
        """
    )
    
    parser.add_argument("input", help="Input image file or directory (if --batch)")
    parser.add_argument("-o", "--output", help="Output image file or directory")
    parser.add_argument("--batch", action="store_true", help="Batch process directory")
    parser.add_argument("--pattern", default="*.png", help="File pattern for batch mode (default: *.png)")
    parser.add_argument("--width", type=int, default=256, help="Output width (default: 256)")
    parser.add_argument("--height", type=int, default=256, help="Output height (default: 256)")
    parser.add_argument("--resampling", 
                       choices=["lanczos", "bilinear", "bicubic", "nearest"],
                       default="lanczos",
                       help="Resampling filter (default: lanczos)")
    
    args = parser.parse_args()
    
    # Map resampling method
    resampling_map = {
        "lanczos": Image.LANCZOS,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "nearest": Image.NEAREST
    }
    resampling = resampling_map[args.resampling]
    
    size = (args.width, args.height)
    
    try:
        if args.batch:
            batch_downscale(args.input, args.output, args.pattern, size)
        else:
            downscale_image(args.input, args.output, size, resampling)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()