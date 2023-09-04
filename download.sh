for year in {1983..2016}; do
    #url="https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0.${year}.days_p05.nc"
    url="https://data.chc.ucsb.edu/products/CHIRTSdaily/v1.0/africa_netcdf_p05/Tmin.${year}.nc"
    downloaded_file="Tmin.${year}.nc"
    sliced_file="chirts.Tmin.${year}.WA.days_p05.nc"

    # Download the file with longer wait time (e.g., 1 hour) and more retries
    wget --waitretry=3600 --tries=999 "$url"

    # Check if the file is downloaded successfully
    if [ -e "$downloaded_file" ]; then
        # Slice the downloaded file
        cdo sellonlatbox,-22,22,3,27 "$downloaded_file" "$sliced_file"
        
        # Remove the downloaded file after slicing
        rm "$downloaded_file"
    else
        echo "Failed to download $downloaded_file"
    fi
done
### Count the number of .nc files in the directory  and Check if the count is equal to 33

if [ -e $(find . -maxdepth 1 -name "chirts.${dt}*.nc" | wc -l) -eq 33 ]; then
    
    for dt in Tmax Tmin; do

        cdo mergetime *${dt}.nc merged_${dt}.nc
    done
    
   fi
# python3 calculation_hwn.py
exit
