import cv2 as cv
import glob as glob
import os

class patcher:

    names = None
    patches = None
    
    #TODO make fuctionality for single image and multiple images
    #TODO check dir_name, do we need to save if just predicting
    #remeber that path can go to either image or dir
    #Last change is dir_name = none
    def __init__(self, path, patch_dim, dir_name = None, gen_data=None):
        if gen_data == None:
            [self.patches, self.names] = self.patch_single_img(path, patch_dim, dir_name)
        else:
            self.make_patches(path, patch_dim, dir_name)

    def read_img(self, path):
        return cv.imread(path)

    def get_bounds(self, img, patch_dim, base_name):
        num_row = int(img.shape[0]/patch_dim)
        num_col = int(img.shape[1]/patch_dim)

        bounds = list()
        names = list()

        for i in range(0,num_col-1):
            for j in range(0, num_row-1):
                patch = [j*patch_dim, (j+1)*patch_dim, i*patch_dim, (i+1)*patch_dim]
                bounds.append(patch)
                name = self.make_name(base_name, j, i, num_row,num_col)
                names.append(name)
        
        return [bounds, names]

    def patch_single_img(self, path, patch_dim, dir_name, basename=None):
        
        if basename == None:
            basename = self.get_basename(path)
        
        img = self.read_img(path)
        [bounds, names] = self.get_bounds(img, patch_dim, basename)
        [bounds_edge, names_edge] = self.get_edge_bounds(img, patch_dim, basename)

        bounds = bounds + bounds_edge
        names = names + names_edge

        patches = list()

        for (bound, name) in zip(bounds, names):
            patch = self.crop_image(img, bound)
            #TODO do we need this for single image
            patches.append(patch)
        
            #self.save_patch(patch, name, dir_name)
    
        return patches, names
            
    def make_patches(self, path, patch_dim, dir_name):

        files = self.read_files(path)
        print(files)
        basenames = self.get_basenames(files)
        
        #for each image
        for (fn, basename) in zip(files, basenames):
            self.patch_single_img(fn, patch_dim, dir_name, basename)

    def save_patch(self, patch, name, dir_name):
        os.mkdir(dir_name) if not os.path.isdir(dir_name) else None
        save_name = dir_name + "/" + name
        cv.imwrite(save_name, patch)
               
    def get_edge_bounds(self, img, patch_dim, base_name):
        num_row = int(img.shape[0]/patch_dim)
        num_col = int(img.shape[1]/patch_dim)
        
        bounds = list()
        names = list()

        #Bottom row
        for i in range(0, num_col-1):
            patch = [img.shape[0]-patch_dim, img.shape[0], i*patch_dim, (1+i)*patch_dim]
            bounds.append(patch)
            name = self.make_name(base_name, num_row, i, num_row, num_col)
            names.append(name)
        
        #Rightmost column
        for i in range(0, num_row-1):
            patch = [i*patch_dim, (i+1)*patch_dim, img.shape[1]-patch_dim, img.shape[1]]
            bounds.append(patch)
            name = self.make_name(base_name, i, num_col-1, num_row, num_col)
            names.append(name)
        
        #Bottom rightmost/Corner patch
        patch = [img.shape[0]-patch_dim, img.shape[0], img.shape[1]-patch_dim, img.shape[1]]
        bounds.append(patch)
        name = self.make_name(base_name, num_row, num_col-1, num_row, num_col)
        names.append(name)
        
        return [bounds, names]
    
    #Crops the image and returns the patch
    def crop_image(self, img, bound):
        #bounds = [y_min, y_max, x_min, x_max]
        
        # img[y:y+h, x:x+w]
        crop_img = img[bound[0]:bound[1], bound[2]:bound[3]]
        return crop_img
    
    #TODO change to pyplot, so kernel doesnt hang
    def display_img(self, img, bound):
        cv.imshow("Displayed img" + str(bound), img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    #Uses parameters to set a unique name and labels with bot or top row
    def make_name(self, base_name, pos_y, pos_x, num_row, num_col):
        name = os.path.splitext(base_name)

        if(pos_y == 0):
            name = name[0] + "_y" + str(pos_y) + "_x" + str(pos_x) + "_top" + name[1]
        elif(pos_y == num_row):
            name = name[0] + "_y" + str(pos_y) + "_x" + str(pos_x) + "_bot" + name[1]
        else:
            name = name[0] + "_y" + str(pos_y) + "_x" + str(pos_x) + name[1]
        
        return name

    def read_files(self, path):
        return glob.glob(path +"/*")

    def get_images(self, files):
        images = list()
        for file in files:
            images.append(self.read_img(file))
        return images
    
    def get_basenames(self, files):
        basenames = list()
        for file in files:
            basenames.append(os.path.basename(file))
        return basenames

    def get_basename(self, path):
        return os.path.basename(path)

    def get_border_patches(self, test):
        size = len(self.names)
        border_patches = list()
        border_names = list()
        
        for i in range(0, size):
            if "top" in self.names[i] or "bot" in self.names[i]:
                border_patches.append(self.patches[i])
                border_names.append(self.names[i])
        
        return border_patches, border_names

    
def main():
    import cv2 as cv
    path = "/home/sina/Documents/abb/refined_data/non_transitions/"
    dir_name = "patch_test_2"
    p = patcher(path, 150, dir_name, gen_data="gen")

if __name__ == "__main__":
    main()

