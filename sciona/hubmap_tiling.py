"""Source raw-tile geometry over explicit decoded RGB and binary mask arrays."""
import numpy as np


class RawTiles:
    """Match source padding and shifted grids, including fully padded tiles.

    Divisible dimensions receive one full tile of padding in the unshifted grid.
    The source's two training grids use offsets0 and tile_size//2. This class
    preserves the general nonnegative offset rule and returns independent arrays.
    """
    def __init__(self,image_rgb,mask,*,tile_size=1024,shift_h=0,shift_w=0):
        image_rgb=np.asarray(image_rgb);mask=np.asarray(mask)
        if image_rgb.ndim!=3 or image_rgb.shape[2]!=3 or image_rgb.dtype!=np.uint8 or not image_rgb.size:
            raise ValueError('Nonempty decoded RGB uint8 image required')
        if mask.shape!=image_rgb.shape[:2] or mask.dtype!=np.uint8 or not np.isin(mask,[0,1]).all():
            raise ValueError('Aligned binary uint8 mask required')
        if isinstance(tile_size,bool) or not isinstance(tile_size,int) or tile_size<1:
            raise ValueError('Positive integer tile size required')
        if any(isinstance(x,bool) or not isinstance(x,int) or not 0<=x<tile_size for x in [shift_h,shift_w]):
            raise ValueError('Integer offsets inside one tile required')
        self.image=image_rgb;self.mask=mask;self.size=tile_size
        self.shift_h=shift_h;self.shift_w=shift_w
        h,w=mask.shape
        self.num_h=(h+tile_size-h%tile_size)//tile_size-int(h%tile_size<shift_h)
        self.num_w=(w+tile_size-w%tile_size)//tile_size-int(w%tile_size<shift_w)

    def __len__(self):
        return self.num_h*self.num_w

    def __getitem__(self,index):
        if isinstance(index,bool) or not isinstance(index,(int,np.integer)) or not 0<=index<len(self):
            raise IndexError('Tile index outside source grid')
        y=(index//self.num_w)*self.size+self.shift_h
        x=(index%self.num_w)*self.size+self.shift_w
        h=min(self.size,self.mask.shape[0]-y);w=min(self.size,self.mask.shape[1]-x)
        image=np.zeros((self.size,self.size,3),dtype=np.uint8)
        mask=np.zeros((self.size,self.size),dtype=np.uint8)
        image[:h,:w]=self.image[y:y+h,x:x+w]
        mask[:h,:w]=self.mask[y:y+h,x:x+w]
        return image,mask
