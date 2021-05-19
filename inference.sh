cd DCNv2_latest && ./make.sh
cd ../src && python inference.py --config $1
