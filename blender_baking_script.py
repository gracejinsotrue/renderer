import bpy
import bmesh
import os

def check_and_fix_uvs():
    """Check if object has UVs and create them if missing"""
    obj = bpy.context.active_object
    
    if not obj or obj.type != 'MESH':
        print("ERROR: No mesh object selected! Please select your model first.")
        return False
        
    print(f"Checking UVs for object: {obj.name}")
    
    # Check if UV map exists
    if not obj.data.uv_layers:
        print("❌ No UV map found. Creating Smart UV Project...")
        
        # Enter edit mode
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Select all
        bpy.ops.mesh.select_all(action='SELECT')
        
        # Smart UV Project
        bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.02)
        
        # Back to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        print("✅ UV unwrapping complete!")
        
    else:
        print("✅ UV map already exists!")
        
    return True

def bake_and_save_tga(bake_type, filename_suffix):
    """Bake texture and save as TGA"""
    
    print(f"\n🔥 Starting bake: {bake_type}")
    
    obj = bpy.context.active_object
    
    # Ensure object has a material
    if not obj.data.materials:
        print("📝 Creating new material...")
        mat = bpy.data.materials.new(name="BakeMaterial")
        mat.use_nodes = True
        obj.data.materials.append(mat)
    
    mat = obj.data.materials[0]
    
    # Enable nodes if not already
    if not mat.use_nodes:
        mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    
    # Create new image
    img = bpy.data.images.new(f"baked_{bake_type}", 1024, 1024)
    print(f"📷 Created 1024x1024 image: baked_{bake_type}")
    
    # Add image texture node
    img_node = nodes.new('ShaderNodeTexImage')
    img_node.image = img
    img_node.select = True
    nodes.active = img_node
    
    # Set render engine to Cycles (required for baking)
    bpy.context.scene.render.engine = 'CYCLES'
    
    print(f"⚙️  Bake settings configured for {bake_type}")
    
    # Bake using the correct method for newer Blender versions
    print("🔥 Baking... (this may take a moment)")
    
    # The bake operation - this works with all Blender versions
    if bake_type == 'DIFFUSE':
        bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})
    elif bake_type == 'NORMAL': 
        bpy.ops.object.bake(type='NORMAL')
    elif bake_type == 'ROUGHNESS':
        bpy.ops.object.bake(type='ROUGHNESS')
    
    # vustom save folder rn
    save_folder = r"C:\Users\gjin3\Desktop\tinyrenderer\obj"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"📁 Created directory: {save_folder}")
    
    # Save as TGA
    filename = f"{obj.name}{filename_suffix}.tga"
    filepath = os.path.join(save_folder, filename)
    
    img.filepath_raw = filepath
    img.file_format = 'TARGA'
    img.save()
    
    print(f"💾 SAVED: {filepath}")
    return filepath

def main():
    """Main function to bake all textures"""
    print("=" * 50)
    print("🚀 STARTING TEXTURE BAKING PROCESS")
    print("=" * 50)
    
    # Check if we have a selected object
    if not bpy.context.active_object:
        print("❌ ERROR: Please select a mesh object first!")
        print("   1. Click on your model in the 3D viewport")
        print("   2. Make sure it's highlighted/selected")
        print("   3. Run this script again")
        return
    
    # Check and fix UVs
    if not check_and_fix_uvs():
        return
    
    print("\n🎯 Starting texture baking...")
    
    # Bake all three texture types
    try:
        diffuse_path = bake_and_save_tga('DIFFUSE', '_diffuse')
        normal_path = bake_and_save_tga('NORMAL', '_nm') 
        spec_path = bake_and_save_tga('ROUGHNESS', '_spec')
        
        print("\n" + "=" * 50)
        print("🎉 SUCCESS! All textures baked:")
        print(f"   Diffuse:  {diffuse_path}")
        print(f"   Normal:   {normal_path}")
        print(f"   Specular: {spec_path}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ ERROR during baking: {str(e)}")
        print("   Make sure your object is a mesh and has proper geometry")

# Run the main function
if __name__ == "__main__":
    main()
