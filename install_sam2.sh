git submodule add https://github.com/facebookresearch/sam2
cd sam2
pip install -e .
pip install rasterio
pip install opencv-python
mv ft-sam2.py sam2/