# python setup.py install
# mv build/*/*.so ./
# rm -rf build/

# in order to setup with python3 use this
cython3 bbox.pyx
cython3 cython_nms.pyx
cython3 gpu_nms.pyx
python3 setup.py build_ext --inplace
mv utils/* ./
rm -rf build
rm -rf utils