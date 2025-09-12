#
# Download a single variable from s3://nex-gddp-cmip6/
# - assumes index_v2.0_md5.txt is present:
#   aws s3 cp --no-sign-request s3://nex-gddp-cmip6/index_v2.0_md5.txt .
#
VAR="pr"
START_YEAR=1984
END_YEAR=2090

# Subset the index file (range of years, selected variable)
rm -f "${VAR}.md5.txt"
for YEAR in $(seq $START_YEAR $END_YEAR); do
    # set up md5 checksum file
    cat index_v2.0_md5.txt | grep "/${VAR}/" | grep "_${YEAR}_v2" | sort >> "${VAR}.md5.txt"
done

# Download
rm -f "${VAR}.to_download.txt"
touch "${VAR}.to_download.txt"
for FILE in $(cat "${VAR}.md5.txt" | cut -d ' ' -f 3); do
    if [ -f $FILE ]; then
        continue
    else
        echo $FILE >> "${VAR}.to_download.txt"
    fi
done

# Echo download command without running
#cat "${VAR}.to_download.txt" | parallel --bar echo "aws s3 cp --quiet --no-sign-request s3://nex-gddp-cmip6/{} {}"
# Run download
cat "${VAR}.to_download.txt" | parallel --bar aws s3 cp --quiet --no-sign-request s3://nex-gddp-cmip6/{} {}

# Checksum
md5sum -c --quiet "${VAR}.md5.txt"
