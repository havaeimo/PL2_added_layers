import theano
import theano.printing as printing
from pylearn2.models.mlp import *

class ConvmixLayer(Layer):
    """
    A wrapper around a different layer that flattens
    the original layer's output.

    The cost works by unflattening the target and then
    calling the wrapped Layer's cost.

    This is mostly intended for use with CompositeLayer as the wrapped
    Layer, and is mostly useful as a workaround for theano not having
    a TupleVariable with which to represent a composite target.

    There are obvious memory, performance, and readability issues with doing
    this, so really it would be better for theano to support TupleTypes.

    See pylearn2.sandbox.tuple_var and the theano-dev e-mail thread
    "TupleType".

    Parameters
    ----------
    raw_layer : WRITEME
        WRITEME
    """

    def __init__(self, raw_layer):
        super(ConvmixLayer, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.layer_name = raw_layer.layer_name

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.raw_layer.set_input_space(space)
        sp1 = self.raw_layer.get_output_space().components[0]
        sp2 = self.raw_layer.get_output_space().components[1]

        nb_channels1 = sp1.num_channels
        nb_channels2 = sp2.num_channels
        
        self.output_space = Conv2DSpace(shape=sp1.shape,
                                             num_channels=nb_channels1+nb_channels2,
                                         axes=('c', 0, 1, 'b')) 

    @wraps(Layer.get_input_space)
    def get_input_space(self):
        return self.raw_layer.get_input_space()

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        return self.raw_layer.get_monitoring_channels(data)

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        return self.raw_layer.get_layer_monitoring_channels(
            state_below=state_below,
            state=state,
            targets=targets
        )


    @wraps(Layer.get_monitoring_data_specs)
    def get_monitoring_data_specs(self):
        return self.raw_layer.get_monitoring_data_specs()

    @wraps(Layer.get_params)
    def get_params(self):
        return self.raw_layer.get_params()

    @wraps(Layer.get_weights)
    def get_weights(self):
        return self.raw_layer.get_weights()

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):
        return self.raw_layer.get_weight_decay(coeffs)

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):
        return self.raw_layer.get_l1_weight_decay(coeffs)

    @wraps(Layer.set_batch_size)
    def set_batch_size(self, batch_size):
        self.raw_layer.set_batch_size(batch_size)

    @wraps(Layer.censor_updates)
    def censor_updates(self, updates):
        self.raw_layer.censor_updates(updates)

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):
        return self.raw_layer.get_lr_scalers()

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        raw = self.raw_layer.fprop(state_below)

        #return self.raw_layer.get_output_space().format_as(raw,
        #                                                   self.output_space)
        return theano.tensor.concatenate(raw, axis=0)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        raw_space = self.raw_layer.get_output_space()
        target_space = self.output_space
        raw_Y = target_space.format_as(Y, raw_space)

        if isinstance(raw_space, CompositeSpace):
            # Pick apart the Join that our fprop used to make Y_hat
            assert hasattr(Y_hat, 'owner')
            owner = Y_hat.owner
            assert owner is not None
            assert str(owner.op) == 'Join'
            # first input to join op is the axis
            raw_Y_hat = tuple(owner.inputs[1:])
        else:
            # To implement this generally, we'll need to give Spaces an
            # undo_format or something. You can't do it with format_as
            # in the opposite direction because Layer.cost needs to be
            # able to assume that Y_hat is the output of fprop
            raise NotImplementedError()
        raw_space.validate(raw_Y_hat)

        return self.raw_layer.cost(raw_Y, raw_Y_hat)

    @wraps(Layer.set_mlp)
    def set_mlp(self, mlp):

        super(ConvmixLayer, self).set_mlp(mlp)
        self.raw_layer.set_mlp(mlp)

    @wraps(Layer.get_weights)
    def get_weights(self):

        return self.raw_layer.get_weights()

    @wraps(Layer.dropout_fprop)
    def dropout_fprop(self, state_below, default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.,
                      input_scales=None, per_example=True, theano_rng=None):

        if theano_rng is None:
            theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

        raw = self.raw_layer.dropout_fprop(
                      state_below,
                      default_input_include_prob=default_input_include_prob,
                      input_include_probs=input_include_probs,
                      default_input_scale=default_input_scale,
                      input_scales=input_scales,
                      per_example=per_example,
                      theano_rng=theano_rng
                  )

        return theano.tensor.concatenate(raw, axis=0)



class ConvaddLayer(Layer):
    """
    A wrapper around a different layer that flattens
    the original layer's output.

    The cost works by unflattening the target and then
    calling the wrapped Layer's cost.

    This is mostly intended for use with CompositeLayer as the wrapped
    Layer, and is mostly useful as a workaround for theano not having
    a TupleVariable with which to represent a composite target.

    There are obvious memory, performance, and readability issues with doing
    this, so really it would be better for theano to support TupleTypes.

    See pylearn2.sandbox.tuple_var and the theano-dev e-mail thread
    "TupleType".

    Parameters
    ----------
    raw_layer : WRITEME
        WRITEME
    """

    def __init__(self, raw_layer):
        super(ConvaddLayer, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.layer_name = raw_layer.layer_name

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.raw_layer.set_input_space(space)
        sp1 = self.raw_layer.get_output_space().components[0]
        sp2 = self.raw_layer.get_output_space().components[1]
        nb_channels1 = sp1.num_channels
        nb_channels2 = sp2.num_channels
        assert sp1.shape == sp2.shape
        assert nb_channels1 == nb_channels2

        self.output_space = Conv2DSpace(shape=sp1.shape,
                                             num_channels=nb_channels2,
                                         axes=('c', 0, 1, 'b'))

    @wraps(Layer.get_input_space)
    def get_input_space(self):
        return self.raw_layer.get_input_space()

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        return self.raw_layer.get_monitoring_channels(data)

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        return self.raw_layer.get_layer_monitoring_channels(
            state_below=state_below,
            state=state,
            targets=targets
            )

    @wraps(Layer.get_monitoring_data_specs)
    def get_monitoring_data_specs(self):
        return self.raw_layer.get_monitoring_data_specs()

    @wraps(Layer.get_params)
    def get_params(self):
        return self.raw_layer.get_params()

    @wraps(Layer.get_weights)
    def get_weights(self):
        return self.raw_layer.get_weights()

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):
        return self.raw_layer.get_weight_decay(coeffs)

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):
        return self.raw_layer.get_l1_weight_decay(coeffs)

    @wraps(Layer.set_batch_size)
    def set_batch_size(self, batch_size):
        self.raw_layer.set_batch_size(batch_size)

    @wraps(Layer.censor_updates)
    def censor_updates(self, updates):
        self.raw_layer.censor_updates(updates)

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):
        return self.raw_layer.get_lr_scalers()

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        raw = self.raw_layer.fprop(state_below)
        output = raw[0] + raw[1]

        #return self.raw_layer.get_output_space().format_as(raw,
        #                                                   self.output_space)
        return output

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        raw_space = self.raw_layer.get_output_space()
        target_space = self.output_space
        raw_Y = target_space.format_as(Y, raw_space)

        if isinstance(raw_space, CompositeSpace):
            # Pick apart the Join that our fprop used to make Y_hat
            assert hasattr(Y_hat, 'owner')
            owner = Y_hat.owner
            assert owner is not None
            assert str(owner.op) == 'Join'
            # first input to join op is the axis
            raw_Y_hat = tuple(owner.inputs[1:])
        else:
            # To implement this generally, we'll need to give Spaces an
            # undo_format or something. You can't do it with format_as
            # in the opposite direction because Layer.cost needs to be
            # able to assume that Y_hat is the output of fprop
            raise NotImplementedError()
        raw_space.validate(raw_Y_hat)

        return self.raw_layer.cost(raw_Y, raw_Y_hat)

    @wraps(Layer.set_mlp)
    def set_mlp(self, mlp):

        super(ConvaddLayer, self).set_mlp(mlp)
        self.raw_layer.set_mlp(mlp)

    @wraps(Layer.get_weights)
    def get_weights(self):

        return self.raw_layer.get_weights()
    @wraps(Layer.dropout_fprop)
    def dropout_fprop(self, state_below, default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.,
                      input_scales=None, per_example=True, theano_rng=None):

        if theano_rng is None:
            theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

        raw = self.raw_layer.dropout_fprop(
                      state_below,
                      default_input_include_prob=default_input_include_prob,
                      input_include_probs=input_include_probs,
                      default_input_scale=default_input_scale,
                      input_scales=input_scales,
                      per_example=per_example,
                      theano_rng=theano_rng
                  )

        return raw[0] + raw[1]

