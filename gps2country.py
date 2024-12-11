import os, sys, exiftool, pickle, shutil

import geopy
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def gps2country(lat, lon):
    locator = geopy.Nominatim(user_agent="myGeocoder")
    location = locator.reverse((lat, lon), language='en')
    return location.raw['address']['country']


if __name__ == '__main__':
    
    im2gps_meta = pickle.load(open('./datasets/metadata_im2gps3k.pkl', 'rb'))

    img2gps_path = './datasets/im2gps3ktest'
    target_path = './datasets/img2country3ktest'

    # for foldername, subfolders, filenames in os.walk(img2gps_path):
    #     for filename in filenames:
    #         if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
    #             continue
    #         # Get the GPS info from the image
    #         img_path = os.path.join(foldername, filename)
    #         img_id = filename.split('.')[0].split('_')[0]

    #         if img_id in im2gps_meta:
    #             gps_info = im2gps_meta[img_id]
    #             lat = gps_info['lat']
    #             lon = gps_info['lon']
    #             country = gps2country(lat, lon)
    #             os.makedirs(os.path.join(target_path, country), exist_ok=True)
    #             shutil.copyfile(img_path, os.path.join(target_path, country, filename))

    all_filenames = []
    for foldername, subfolders, filenames in os.walk(target_path):
        for filename in filenames:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            all_filenames.append(filename)
    print(len(all_filenames))
            
    count = 0
    for foldername, subfolders, filenames in os.walk(img2gps_path):
        for filename in filenames:
            if not filename in all_filenames:
                print(filename)
                # get the GPS info from the image
                img_path = os.path.join(foldername, filename)
                img_id = filename.split('.')[0].split('_')[0]
                if img_id in im2gps_meta:
                    gps_info = im2gps_meta[img_id]
                    lat = gps_info['lat']
                    lon = gps_info['lon']
                    country = gps2country(lat, lon)
                    print(lat, lon)
                    print(country)
            else:
                count += 1
    
    print(count)
    print(len(im2gps_meta.keys()))
            

    