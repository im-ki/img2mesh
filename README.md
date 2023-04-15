# img2mesh
Image with human face to 3DMM. 

This code is based on https://github.com/sicxu/Deep3DFaceRecon_pytorch

------------------------------------------------------------------------
To run the code:

Step 1: Create a conda environment and install some necessary packages.
```console
conda create --name img2mesh
conda install -c conda-forge pytorch numpy scipy pillow cython trimesh
conda activate img2mesh
```

Step 2: Put the correct BFM and Network model file to BFM/ and checkpoints/.

TO BE COMPLETED !!!!!!

Step 3: Compile the renderer.

```console
cd renderer
python setup.py build_ext --inplace
cd ..
```

Step 4: Put some images into the 'test_data' folder and run the code to generate 3DMM vectors and rendered images.
```console
python demo.py
```
And the output images, shape and texture vectors are in the 'test_data/output' folder.

Step 5: Input the following command:

If you want to generate a mesh file with texture, then run the following command:
```console
python3 generate_ply.py -p test_data/output -i 000031_id.npy -e vd070_exp.npy -t vd070_tex.npy
```
else run the following:
```console
python3 generate_ply.py -p test_data/output -i 000031_id.npy -e vd070_exp.npy
```

Then the output ply files will be in the 'test_data/output' folder.
