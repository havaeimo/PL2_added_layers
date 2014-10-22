import theano
import theano.printing as printing
from pylearn2.models.mlp import *
#import theano 

class Softmax_multidim(Layer):
    """
    .. todo::

        WRITEME (including parameters list)

    Parameters
    ----------
    n_classes : WRITEME
    layer_name : WRITEME
    irange : WRITEME
    istdev : WRITEME
    sparse_init : WRITEME
    W_lr_scale : WRITEME
    b_lr_scale : WRITEME
    max_row_norm : WRITEME
    no_affine : WRITEME
    max_col_norm : WRITEME
    init_bias_target_marginals : WRITEME
    """

    def __init__(self, n_classes, layer_name, irange=None,
                 istdev=None,
                 sparse_init=None, W_lr_scale=None,
                 b_lr_scale=None, max_row_norm=None,
                 no_affine=True,
                 max_col_norm=None, init_bias_target_marginals=None):

        super(Softmax_multidim, self).__init__()

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)

        self.__dict__.update(locals())
        del self.self
        del self.init_bias_target_marginals

        assert isinstance(n_classes, py_integer_types)

        

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        self.output_space = VectorSpace(self.input_space.shape[0]*self.input_space.shape[1]*5)
        
        self.desired_space = self.input_space
        if not isinstance(space, Space):
            raise TypeError("Expected Space, got " +
                            str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, Conv2DSpace)
                                   
        desired_dim = self.n_classes
       
        self._params = []        
       


    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)
       
        self.desired_space.validate(state_below)
        assert state_below.ndim == 4
        assert self.mlp.batch_size == 128
        
        Z = state_below
        
        
        e_Z = T.exp(Z - Z.max(axis=0, keepdims=True))
        rval = e_Z /e_Z.sum(axis=0, keepdims=True)

        rval_swaped = rval.dimshuffle(3,1,2,0)

        rval_swaped = rval_swaped.reshape(shape=(self.mlp.batch_size,self.input_space.shape[0]*self.input_space.shape[1]*5),ndim=2)      
        
        
        return rval_swaped

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        z = Y_hat
        
        z = z.reshape(shape=(self.mlp.batch_size*
                                   self.input_space.shape[0]*
                                   self.input_space.shape[1],
                                   self.input_space.num_channels ),ndim=2)
        
        Y = Y.reshape(shape=(self.mlp.batch_size*
                                   self.input_space.shape[0]*
                                   self.input_space.shape[1],
                                   self.input_space.num_channels ),ndim=2)
        
       
        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        # we use sum and not mean because this is really one variable per row
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1
        rval = log_prob_of.mean()

        return  - rval


    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):

        # channels that does not require state information
        rval = OrderedDict()

        if (state_below is not None) or (state is not None):
            
            state = state.reshape(shape=(self.mlp.batch_size*
                                   self.input_space.shape[0]*
                                   self.input_space.shape[1],
                                   self.input_space.num_channels ),ndim=2)

            mx = state.max(axis=1)   

            rval.update(OrderedDict([('mean_max_class', mx.mean()),
                                ('max_max_class', mx.max()),
                                ('min_max_class', mx.min())]))

            assert  targets is not None
            targets = targets.reshape(shape=(self.mlp.batch_size*
                                   self.input_space.shape[0]*
                                   self.input_space.shape[1],
                                   self.input_space.num_channels ),ndim=2)

                          
            
            y_hat = T.argmax(state, axis=1)
            y = T.argmax(targets, axis=1)
            
            #state = printing.Print('state')(state)
            misclass = T.neq(y, y_hat).mean()
            misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass
            
            state = state.reshape(shape=(self.mlp.batch_size,
                                   self.input_space.shape[0]*
                                   self.input_space.shape[1]*
                                   self.input_space.num_channels ),ndim=2)
            targets = targets.reshape(shape=(self.mlp.batch_size,
                                   self.input_space.shape[0]*
                                   self.input_space.shape[1]*
                                   self.input_space.num_channels ),ndim=2)

            rval['nll'] = self.cost(Y_hat=state, Y=targets)

        return rval


    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        return 0



    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        return 0

