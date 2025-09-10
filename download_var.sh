#
# Download a single variable from s3://nex-gddp-cmip6/
# - assumes index_v2.0_md5.txt is present
#   aws s3 cp --no-sign-request s3://nex-gddp-cmip6/index_v2.0_md5.txt .
#
VAR="tasmax"
START_YEAR=1984
END_YEAR=2090

# Subset the index file (range of years, tasmax variable)
rm -f "${VAR}.md5.txt"
for YEAR in $(seq $START_YEAR $END_YEAR); do
    # set up md5 checksum file
    cat index_v2.0_md5.txt | grep "/${VAR}/" | grep "_${YEAR}_v2" | sort >> "${VAR}.md5.txt"
done

# Download
for YEAR in $(seq $START_YEAR $END_YEAR); do
    # Echo download command without running                        vvvv
    cat "${VAR}.md5.txt" | grep $YEAR | cut -d ' ' -f 3 | parallel echo "aws s3 cp --no-sign-request s3://nex-gddp-cmip6/{} {}"
    # Run download                                                   vvvvvvvvv
    # cat "${VAR}.md5.txt" | grep $YEAR | cut -d ' ' -f 3 | parallel aws s3 cp --no-sign-request s3://nex-gddp-cmip6/{} {}
done

# Checksum
md5sum -c "${VAR}.md5.txt"
