from collections import namedtuple

Params = namedtuple("Config", [
  'img_rows',
  'img_cols',
  'lr',
  'optimizer',
  'batch_size',
  'epoch_size',
  'CLAHE',
  'nb_epoch',
  'predict_batch_size',
  'CROP',
  'Flip',
  'lighting',
  'affine',
  'randcrop',
  'perspective',
  'dbg',
  'save_images',
  'weights',
  'orig_path',
  'data_path',
  'data_path_test',
  'data_path_eval',
  'result_dir' 

])