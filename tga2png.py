#!/usr/bin/env python3
"""
TGA to PNG Converter
manually parse tga file to python so we can view it like a normal person 
"""

import sys
import os
import struct
from PIL import Image
import argparse

class TGAReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.header = {}
        self.image_data = None
        self.width = 0
        self.height = 0
        self.bpp = 0
        
    def read_header(self, f):
        """read TGA header structure"""
        # a tga ifile is comprised of 18 bits in little endian format
        header_data = f.read(18)
        if len(header_data) < 18:
            raise ValueError("header too short this is not a valid tga")
        # this is tga definition. 1 bit for id, 1 bit for color map, 1 bit for img type, etc etc 
        #https://en.wikipedia.org/wiki/Truevision_TGA
        self.header = {
            'id_length': header_data[0],
            'colormap_type': header_data[1], 
            'image_type': header_data[2],
            'colormap_origin': struct.unpack('<H', header_data[3:5])[0],
            'colormap_length': struct.unpack('<H', header_data[5:7])[0],
            'colormap_depth': header_data[7],
            'x_origin': struct.unpack('<H', header_data[8:10])[0],
            'y_origin': struct.unpack('<H', header_data[10:12])[0],
            'width': struct.unpack('<H', header_data[12:14])[0],
            'height': struct.unpack('<H', header_data[14:16])[0],
            'bpp': header_data[16],
            'image_descriptor': header_data[17]
        }
        
        self.width = self.header['width']
        self.height = self.header['height'] 
        self.bpp = self.header['bpp'] // 8  # convert the bit to byte
        
        print(f"TGA Info: {self.width}x{self.height}, {self.header['bpp']} bpp, type {self.header['image_type']}")
        
    def read_image_data(self, f):
        """Read the actual image pixel data"""
        # skip id field if it is present 
        if self.header['id_length'] > 0:
            f.read(self.header['id_length'])
            
        # skip colormap if present
        if self.header['colormap_type'] == 1:
            colormap_size = self.header['colormap_length'] * (self.header['colormap_depth'] // 8)
            f.read(colormap_size)
        
        image_type = self.header['image_type']
        pixel_count = self.width * self.height
        
        if image_type in [2, 3]:  # it is uncompresseed rbg, or it is colorscale
            expected_size = pixel_count * self.bpp
            self.image_data = f.read(expected_size)
            if len(self.image_data) < expected_size:
                raise ValueError(f"Not enough image data: got {len(self.image_data)}, expected {expected_size}")
                
        elif image_type in [10, 11]:  # RLE compressed
            self.image_data = self.read_rle_data(f, pixel_count)
        else:
            raise ValueError(f"Unsupported TGA image type: {image_type}")
    
    def read_rle_data(self, f, pixel_count):
        """Read RLE compressed image data"""
        data = bytearray()
        pixels_read = 0
        
        while pixels_read < pixel_count:
            try:
                packet_header = f.read(1)[0]
            except:
                raise ValueError("unexpected end of file in RLE data")
                
            if packet_header & 0x80:  # RLE packet
                run_length = (packet_header & 0x7F) + 1
                pixel_data = f.read(self.bpp)
                if len(pixel_data) < self.bpp:
                    raise ValueError("unexpected end of file in RLE pixel data")
                for _ in range(run_length):
                    data.extend(pixel_data)
                pixels_read += run_length
            else:  # Raw packet
                run_length = (packet_header & 0x7F) + 1
                pixel_data = f.read(run_length * self.bpp)
                if len(pixel_data) < run_length * self.bpp:
                    raise ValueError("unexpected end of file in raw pixel data")
                data.extend(pixel_data)
                pixels_read += run_length
                
        return bytes(data)
    
    def to_pil_image(self):
        """Convert TGA data to PIL image using peethon pil library"""
        if self.image_data is None:
            raise ValueError("no image data loaded")
            
        # Convert BGR(A) to RGB(A) format for PIL
        if self.bpp == 3:  # RGB
            # TGA stores as BGR, convert to RGB
            rgb_data = bytearray()
            for i in range(0, len(self.image_data), 3):
                b, g, r = self.image_data[i:i+3]
                rgb_data.extend([r, g, b])
            mode = 'RGB'
            data = bytes(rgb_data)
            
        elif self.bpp == 4:  # RGBA
            # TGA stores as BGRA, convert to RGBA
            rgba_data = bytearray()
            for i in range(0, len(self.image_data), 4):
                b, g, r, a = self.image_data[i:i+4]
                rgba_data.extend([r, g, b, a])
            mode = 'RGBA'
            data = bytes(rgba_data)
            
        elif self.bpp == 1:  # then it is a grayscale
            mode = 'L'
            data = self.image_data
        else:
            raise ValueError(f"unsupported bytes per pixel: {self.bpp}")
        
        # finally create the dam image
        img = Image.frombytes(mode, (self.width, self.height), data)
        
        # imag orientation based on image descriptor
        if not (self.header['image_descriptor'] & 0x20):  # Origin at bottom
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if self.header['image_descriptor'] & 0x10:  # Origin at right
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        return img

# if pil fails because of skill issue, we will just do skill issue
def convert_tga_to_png_robust(tga_path, png_path=None):
    """
    Convert TGA to PNG using manual TGA parsing for better compatibility.
    """
    try:
        #crete the output path if ot specified
        if png_path is None:
            base_name = os.path.splitext(tga_path)[0]
            png_path = f"{base_name}.png"
        
        # try faster pil first
        try:
            with Image.open(tga_path) as img:
                img.save(png_path, "PNG")
                print(f"Converted (PIL): {tga_path} → {png_path}")
                return png_path
        except Exception as pil_error:
            print(f"PIL failed ({pil_error}), trying to parse like a normal prson because what even is pil")
        
        # cringe
        reader = TGAReader(tga_path)
        
        with open(tga_path, 'rb') as f:
            reader.read_header(f)
            reader.read_image_data(f)
        
        # convert to PIL Image and save
        img = reader.to_pil_image()
        img.save(png_path, "PNG")
        
        print(f"onverted (manual): {tga_path} → {png_path}")
        return png_path
        
    except Exception as e:
        print(f"✗ Error converting {tga_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert TGA files to PNG format")
    parser.add_argument("input", help="Input TGA file or directory")
    parser.add_argument("-o", "--output", help="Output PNG file (for single file conversion)")
    parser.add_argument("-r", "--recursive", action="store_true", 
                       help="Recursively convert all TGA files in directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    input_path = args.input
    
    # input handling
    if not os.path.exists(input_path):
        print(f"cringe {input_path} does not exist")
        sys.exit(1)
    
    # Single file conversion
    if os.path.isfile(input_path):
        if not input_path.lower().endswith('.tga'):
            print("cringe input file must have .tga extension")
            sys.exit(1)
        
        result = convert_tga_to_png_robust(input_path, args.output)
        if result:
            print("this shit works")
        else:
            sys.exit(1)
    
    # directory conversion
    elif os.path.isdir(input_path):
        tga_files = []
        
        if args.recursive:
            # find all TGA files recursively
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith('.tga'):
                        tga_files.append(os.path.join(root, file))
        else:
            # find TGA files in the specified directory only
            for file in os.listdir(input_path):
                if file.lower().endswith('.tga'):
                    tga_files.append(os.path.join(input_path, file))
        
        if not tga_files:
            print(f"No TGA files found in {input_path}")
            sys.exit(1)
        
        print(f"Found {len(tga_files)} TGA file(s)")
        
        successful = 0
        for tga_file in tga_files:
            if convert_tga_to_png_robust(tga_file):
                successful += 1

if __name__ == "__main__":
    # for no arguments
    if len(sys.argv) == 1:
        print("TGA to PNG Converter")
        print("\nUsage examples:")
        print("  python tga2png.py output.tga                    # Convert single file")
        print("  python tga2png.py output.tga -o myimage.png     # Convert with custom output name") 
        print("  python tga2png.py ./images/                     # Convert all TGA files in directory")
        print("  python tga2png.py ./project/ -r                 # Convert all TGA files recursively")
        print("  python tga2png.py output.tga -v                 # Verbose output")
        print("\nRequired: pip install Pillow")
        sys.exit(0)
    
    main()