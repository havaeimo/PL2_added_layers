import numpy as np
#from matplotlib.pyplot import imsave
import numpy
import argparse
from lisa_brats.brains import BrainSet
from lisa_brats.misc.preprocessing import standardize_nonzeros
#from lisa_brats.patches import all_patches
#from lisa_brats.utils import probability_labels_add_borders
import cPickle
import theano
import os
import os.path
#import ipdb
#from scipy.misc import imsave

"""
Output:
brains x label distr x slice x row x column
"""
def all_patches(padded_brain,i,predict_patchsize,obs_patchsize,num_channels):
    
    image = padded_brain[i]
    ishape_h , ishape_w = padded_brain.shape[1:3]
    #ipdb.set_trace()
    #ipdb.set_trace()
    half_obs_patchsize = obs_patchsize/2
    half_predict_patchsize = predict_patchsize/2
    
    secondhalf_obs_patchsize = obs_patchsize - obs_patchsize/2
    secondhalf_predict_patchsize = predict_patchsize -  predict_patchsize/2

    extended_image = np.zeros((ishape_h+obs_patchsize-predict_patchsize,ishape_w+obs_patchsize-predict_patchsize,num_channels))

    extended_image[half_obs_patchsize - half_predict_patchsize   : -(secondhalf_obs_patchsize - secondhalf_predict_patchsize) ,half_obs_patchsize - half_predict_patchsize  : -(secondhalf_obs_patchsize - secondhalf_predict_patchsize)]= image
    num_patches_rows = ishape_h // predict_patchsize
    num_patches_cols = ishape_w // predict_patchsize
    
    list_patches = np.zeros((num_patches_cols*num_patches_rows, obs_patchsize, obs_patchsize, num_channels))
    index = 0
    h_range = np.arange(obs_patchsize/2,ishape_h+obs_patchsize/2,predict_patchsize)
    #h_range = h_range[:-1]
    v_range = np.arange(obs_patchsize/2,ishape_w+obs_patchsize/2,predict_patchsize)
    #v_range = v_range[:-1]
    #ipdb.set_trace()
    for index_h in h_range:
        for index_w in v_range:
            patch_brian = extended_image[index_h-half_obs_patchsize : index_h+secondhalf_obs_patchsize 
,index_w-half_obs_patchsize  : index_w+secondhalf_obs_patchsize,:]
            #if patch_brian.shape == (38,29,4):
            #ipdb.set_trace()
             
            list_patches[index,:,:,:] = patch_brian
            index += 1
    #ipdb.set_trace()
    assert index == num_patches_rows*num_patches_cols
    return list_patches       

def generate_prediction_for_data(data, fprop, batch_size = 128):
    """
    Parameters
    ----------
    data: ndarray, shape (n_patches, n_row, n_col, n_channels)
    fprop: the thean function

    Returns:
    -------
    predictions with fprop.
        ndarray, shape (n_patches, n_classes)
    """
    batches = int(numpy.ceil(data.shape[0] / float(batch_size)))
    results = []
    for b in xrange(batches):
        batch = data[b * batch_size:(b + 1) * batch_size]
        batch = batch.swapaxes(0, 3).copy()
        num_samples = batch.shape[-1]         
        if num_samples < batch_size:
            buffer_batch = np.zeros((batch.shape[0],batch.shape[1],batch.shape[2],batch_size),dtype=np.float32)
            buffer_batch[:,:,:,0:num_samples] = batch
            batch = buffer_batch

        results_batch = fprop(batch)
        #ipdb.set_trace()
        if num_samples < batch_size:
            results_batch = results_batch[0:num_samples,...]

        results.extend(results_batch)
    return results

def generate_prediction_for_brain(brain,predict_patchsize, fprop,obs_patchsize,num_channels):
    """
    Parameters
    ----------
    brain: a brain
    patch_shape: the size of the patches,
    fprop: the theano function
    
    Returns
    --------
    results: ndarray, shape (n_classes, n_r, n_c)
        An array containing the predictions for each class
        and pixel of fprop
    """
    results = []
    slices = b.images.copy()
    #slices = standardize_nonzeros(slices, axis=3) BESURE TO UNCOMMENT THIS----IMP*****
    num_levels = len(slices)
    depth, height , width = slices.shape[0:3]
    #ipdb.set_trace()
    if height % predict_patchsize != 0 :
       ishape_h = height + predict_patchsize - height % predict_patchsize
    else :
       ishape_h = height

    if width % predict_patchsize != 0 :
       ishape_w = width + predict_patchsize - width % predict_patchsize
    else :
       ishape_w = width
    #if ishape_h % 2 != 0 or ishape_w %2 != 0 :
    #ipdb.set_trace()
       
   
    #assert ishape_h % 2 == 0 and ishape_w %2 == 0
    padding = (ishape_h - height, ishape_w - width)
    #padding = (predict_patchsize - height % predict_patchsize ,predict_patchsize - width % predict_patchsize) 
    padded_brain = np.zeros((depth, ishape_h, ishape_w,num_channels))
    #ipdb.set_trace()
    padded_height = padded_brain.shape[1]
    padded_width = padded_brain.shape[2]
    padded_brain[:,0+padding[0]/2:padded_height-(padding[0]-padding[0]/2),0+padding[1]/2:padded_width-(padding[1]-padding[1]/2),0:num_channels] = slices
    #padded_brain = slices
    num_patches_rows = ishape_h // predict_patchsize
    num_patches_cols = ishape_w // predict_patchsize
    assert ishape_h % predict_patchsize == 0
    assert ishape_w % predict_patchsize == 0
    #ipdb.set_trace()
    patches_shape = (num_patches_rows, num_patches_cols)
    prediction = np.zeros((5,depth, ishape_h, ishape_w),dtype=np.float32)
    indexx=0
    for z in range(num_levels):
        patches = all_patches(padded_brain, z, predict_patchsize, obs_patchsize, num_channels)
        results = generate_prediction_for_data(numpy.asarray(patches,dtype=np.float32), fprop)
        results = np.asarray(results, dtype=numpy.float32)
        #results = results.T
        for r in range(num_patches_rows):
            for c in range(num_patches_cols):
                index = r*num_patches_cols + c
                id_batch = results[index,...]
                id_batch = id_batch.reshape(predict_patchsize*predict_patchsize,5)
                id_batch = id_batch.T
                id_batch = id_batch.reshape(5,predict_patchsize, predict_patchsize)
                prediction[:,z,predict_patchsize*r:predict_patchsize*(r+1),predict_patchsize*c:predict_patchsize*(c+1)] = id_batch
        #label_prediction = prediction[z,:,:,:].argmax(axis=2)
        #x,y = np.where(label_prediction!=0)
        #print label_prediction[x,y]
        #imsave('slice_%s.png'%str(indexx),label_prediction,vmax=4)   
        #print 'slice_%s.png'%str(indexx)
        #indexx+=1
    #prediction = prediction.swapaxes(0,3)
    adjusted_prediction = prediction[:,:,0+padding[0]/2:padded_height-(padding[0]-padding[0]/2),0+padding[1]/2:padded_width-(padding[1]-padding[1]/2)]
    return adjusted_prediction

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate the DICE score for a BrainSet')
    parser.add_argument('model', type=argparse.FileType('r'),
                        help='A serialized pylearn2 model.')
    parser.add_argument('brain_set', type=str,
                        help='The serialized BrainSet to read.'),
    #parser.add_argument('patch_shape', type=int,
    #                    help='The size of the input patch window.'),
    parser.add_argument('label_patch_shape', type=int,
                        help='The size of the predicted patch window.'), 
    #parser.add_argument('num_channels', type=int,
    #                    help='Number of channels in the dataset.'),
    args = parser.parse_args()

    brain_set = BrainSet.from_path(args.brain_set)
    #import ipdb
    #ipdb.set_trace()
    model = cPickle.load(args.model)
    patch_shrink_size = model.input_space.shape[0] - model.layers[-1].input_space.shape[0]
    patch_shape = args.label_patch_shape + patch_shrink_size

    if (patch_shape % 2) != 0:     
        raise Exception('Oops!  Choose a size for the  predicted patch so that the input patch size to the model would be even')


    num_channels = model.input_space.num_channels
    model.layers[-1].input_space.shape = (args.label_patch_shape, args.label_patch_shape)
    model.layers[-1].desired_space.shape = (args.label_patch_shape, args.label_patch_shape)
    #ipdb.set_trace() 
    X = model.get_input_space().make_theano_batch()
    f = theano.function([X], model.fprop(X))
    #theano.printing.debugprint(f)
    #find a way to get teh shape from the softmax layer for now im dividing by 2 but
    #that needs to be fixed
    fprop_input_shape = model.get_input_space().shape
    obs_patchsize = patch_shape
    #predict_patch_size = (fprop_input_shape[0]/2,fprop_input_shape[1]/2)
    predict_patch_size = args.label_patch_shape
    brains = brain_set.get_brains()
    for name in brains:
        b = brains[name]
        print 'brain ' + name
        fname = name + str('_predictions.npy')
        if os.path.exists(fname):
            print fname + ' exists already. skipping'
            continue
        #ipdb.set_trace()
        prediction = generate_prediction_for_brain(b, predict_patch_size, f,obs_patchsize,num_channels)


        fhandle = open(fname, 'wb+') 
        numpy.save(fhandle, prediction)
        fhandle.close()
        #ipdb.set_trace()
