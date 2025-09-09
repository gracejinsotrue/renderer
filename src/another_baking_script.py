#just for personal use it speeds it up so much
import bpy
import bmesh
import os

def check_and_fix_uvs(obj):
    """Check if object has UVs and create them if missing"""
    if not obj or obj.type != 'MESH':
        print(f"ERROR: {obj.name if obj else 'None'} is not a mesh object!")
        return False
        
    print(f"Checking UVs for object: {obj.name}")
    
    # Check if UV map exists
    if not obj.data.uv_layers:
        print("❌ No UV map found. Creating Smart UV Project...")
        
        # Make sure object is active and selected
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        
        # Enter edit mode
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

def bake_and_save_tga(obj, bake_type, filename_suffix):
    """Bake texture and save as TGA"""
    
    print(f"\n🔥 Starting bake: {bake_type} for {obj.name}")
    
    # Make sure this object is active
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    
    # Ensure object has a material
    if not obj.data.materials:
        print("📝 Creating new material...")
        mat = bpy.data.materials.new(name=f"BakeMaterial_{obj.name}")
        mat.use_nodes = True
        obj.data.materials.append(mat)
    
    mat = obj.data.materials[0]
    
    # Enable nodes if not already
    if not mat.use_nodes:
        mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    
    # Create new image
    img = bpy.data.images.new(f"baked_{obj.name}_{bake_type}", 1024, 1024)
    print(f"📷 Created 1024x1024 image: baked_{obj.name}_{bake_type}")
    
    # Add image texture node
    img_node = nodes.new('ShaderNodeTexImage')
    img_node.image = img
    img_node.select = True
    nodes.active = img_node
    
    # Set render engine to Cycles (required for baking)
    bpy.context.scene.render.engine = 'CYCLES'
    
    print(f"⚙️  Bake settings configured for {bake_type}")
    
    # Bake using the correct method
    print("🔥 Baking... (this may take a moment)")
    
    try:
        if bake_type == 'DIFFUSE':
            bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})
        elif bake_type == 'NORMAL': 
            bpy.ops.object.bake(type='NORMAL')
        elif bake_type == 'ROUGHNESS':
            bpy.ops.object.bake(type='ROUGHNESS')
    except Exception as e:
        print(f"❌ Baking failed for {bake_type}: {str(e)}")
        return None
    
    # Save folder
    save_folder = r"C:\Users\gjin3\Desktop\tinyrenderer\objs"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"📁 Created directory: {save_folder}")
    
    # Save as TGA
    clean_name = obj.name.replace(" ", "_").replace(".", "_")
    filename = f"{clean_name}{filename_suffix}.tga"
    filepath = os.path.join(save_folder, filename)
    
    img.filepath_raw = filepath
    img.file_format = 'TARGA'
    img.save()
    
    print(f"💾 SAVED: {filepath}")
    
    # Clean up the baking node
    nodes.remove(img_node)
    
    return filepath

def export_obj(obj, save_folder):
    """Export object as OBJ"""
    # Make sure this object is active and selected
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    
    clean_name = obj.name.replace(" ", "_").replace(".", "_")
    filename = f"{clean_name}.obj"
    filepath = os.path.join(save_folder, filename)
    
    bpy.ops.export_scene.obj(
        filepath=filepath,
        use_selection=True,
        use_materials=True,
        use_uvs=True,
        use_normals=True
    )
    
    print(f"📦 EXPORTED OBJ: {filepath}")
    return filepath

def main():
    """Main function to bake all textures for all mesh objects"""
    print("=" * 60)
    print("🚀 STARTING BATCH TEXTURE BAKING PROCESS")
    print("=" * 60)
    
    # Get all mesh objects in scene
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    if not mesh_objects:
        print("❌ ERROR: No mesh objects found in scene!")
        return
    
    print(f"Found {len(mesh_objects)} mesh objects to process")
    
    save_folder = r"C:\Users\gjin3\Desktop\tinyrenderer\objs"
    
    for obj in mesh_objects:
        print(f"\n🎯 Processing object: {obj.name}")
        print("-" * 40)
        
        # Check and fix UVs
        if not check_and_fix_uvs(obj):
            print(f"❌ Skipping {obj.name} - UV issues")
            continue
        
        # Bake all three texture types
        try:
            diffuse_path = bake_and_save_tga(obj, 'DIFFUSE', '_diffuse')
            normal_path = bake_and_save_tga(obj, 'NORMAL', '_nm') 
            spec_path = bake_and_save_tga(obj, 'ROUGHNESS', '_spec')
            
            # Export OBJ
            obj_path = export_obj(obj, save_folder)
            
            print(f"\n✅ SUCCESS for {obj.name}:")
            print(f"   Diffuse:  {diffuse_path}")
            print(f"   Normal:   {normal_path}")
            print(f"   Specular: {spec_path}")
            print(f"   OBJ:      {obj_path}")
            
        except Exception as e:
            print(f"❌ ERROR processing {obj.name}: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("🎉 BATCH PROCESSING COMPLETE!")
    print("=" * 60)

# Run the main function
if __name__ == "__main__":
    main()
