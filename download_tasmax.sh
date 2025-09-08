cat index_v2.0_md5.txt | cut -d ' ' -f 3 | grep tasmax | sort | sed 's/^/s3:\/\/nex-gddp-cmip6\//' > tasmax.txt
cat tasmax.txt |  parallel aws s3 cp {} data/
