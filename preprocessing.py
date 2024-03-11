
import sys
import pathlib
import shutil
import PIL
import numpy as np
import tensorflow as tf
from target_dirs import get_blacklist, get_directories






def get_img_from_path(path):
  return PIL.Image.open(str(path))

def get_rgb_img_from_path(path):
  return PIL.Image.open(str(path)).convert('RGB')

def get_tensor_from_img(img):
  return tf.convert_to_tensor(tf.keras.preprocessing.image.img_to_array(img), dtype=tf.float32)




def has_black_background(rgb_img):
  """Returns True if the image background is black, otherwise False
  It assumes the majority of the image's border pixels are the background color.
  """
  top_rows = rgb_img[0,:,:]
  bottom_rows = rgb_img[-1,:,:]
  left_cols = rgb_img[:,0,:]
  right_cols = rgb_img[:,-1,:]
  together = tf.reshape(tf.concat([top_rows, bottom_rows, left_cols, right_cols], axis=0),[-1]).numpy()
  vals, counts = np.unique(together, return_counts=True)
  return vals[np.argmax(counts)] == 0.0


def gather_all_files(filepaths, target_dir):
  destination_dir = pathlib.Path(target_dir)
  destination_dir.mkdir(parents=True, exist_ok=True)

  for count, filepath in enumerate(filepaths):
    if filepath.exists() and filepath.is_file():
      destination_path = destination_dir / filepath.name

      if not destination_path.exists():
        shutil.copy2(filepath, destination_path)
      else:
        num = 0
        dpath_stem = destination_path.stem
        dpath_suffix = destination_path.suffix
        while (destination_dir /pathlib.Path(dpath_stem+str(num)+dpath_suffix)).exists():
          num += 1
        #print(dpath_stem+str(num)+dpath_suffix)
        shutil.copy2(filepath, destination_dir / pathlib.Path(dpath_stem+str(num)+dpath_suffix))
    else:
      print(f"File '{filepath}' does not exist or is not a regular file")

  print(f'Transferred {count+1} files')



def white_bkgd_to_black(img):
    """Converts a non-black background rgba image to a black background rgb image
    """
    t_img = get_tensor_from_img(img)
    rgb = t_img[:,:,:3]
    a = t_img[:,:,3]
    black = rgb * (tf.expand_dims(a, axis=-1) / 255)
    return tf.keras.utils.array_to_img(black)


def convert_bkgd_and_save(filepaths, target_dir):
  """Converts
  """
  destination_dir = pathlib.Path(target_dir)

  for count, filepath in enumerate(filepaths):
    img = PIL.Image.open(str(filepath))
    converted_img = white_bkgd_to_black(img)


    destination_path = destination_dir / filepath.name
    if not destination_path.exists():
      converted_img.save(str(destination_path))
    else:
      num = 0
      dpath_stem = destination_path.stem
      dpath_suffix = destination_path.suffix
      while (destination_dir / pathlib.Path(dpath_stem+str(num)+dpath_suffix)).exists():
          num += 1
      converted_img.save(str(destination_dir / pathlib.Path(dpath_stem+str(num)+dpath_suffix)))
  print(f'Converted and transferred {count+1} files')




def main():
    argc = len(sys.argv)
    if argc != 3:
        print("usage: python preprocessing.py <path to sprites> <target directory>")
        sys.exit()
    data_dir = sys.argv[1]
    if data_dir[-1] == '/':
       data_dir = data_dir[:-1]
    target_dir = sys.argv[2]


    # obtain list of directory paths and list of paths to bad files
    good_dirs = get_directories(pathlib.PosixPath(data_dir))
    blacklist = get_blacklist(data_dir)

    # obtain complete list of paths to .png files in the considered directories
    pokes = []
    for d in good_dirs:
        pokes += list(d.with_suffix('').glob('*.png'))
    #pokes_count = len(pokes)

    #filter blacklisted    
    pokes = [p for p in pokes if p not in blacklist]
   

    #black background pokemon get saved to target_dir
    #white background pokemon are converted to black background and then saved
    black_bkgd_pokes  = [p for p in pokes if has_black_background(get_tensor_from_img(get_rgb_img_from_path(p)))]
    white_bkgd_pokes = [p for p in pokes if not has_black_background(get_tensor_from_img(get_rgb_img_from_path(p)))]

    if len(black_bkgd_pokes) > 0:
        gather_all_files(black_bkgd_pokes, pathlib.PosixPath(target_dir))
    else:
       print("No black background pokémon.")

    convertible_white_bkgd = [p for p in white_bkgd_pokes if PIL.Image.open(p).mode == 'RGBA']
    #print("len(convertible white)", len(convertible_white_bkgd))
    if len(convertible_white_bkgd) > 0:
        convert_bkgd_and_save(convertible_white_bkgd, pathlib.PosixPath(target_dir))
    else:
        print("No non-black background pokémon.")









if __name__ == "__main__":
    main()