class Vectorspace2ConvspaceLayer(Layer):
    """
    used to output of softmax to Conv2DSpace
    """

    def __init__(self, layer_name, size,channels,axes):
        super(Vectorspace2ConvspaceLayer, self).__init__()
        self.__dict__.update(locals())
        del self.self
        if size[0] != size[1] :
            raise ValueError("Vectorspace2ConvspaceLayer: bad shape parameter")

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        """
        Method to reshape the output of multi_softmax layer. the vector sapec is in shape ('b,'shape[0]*shape[1]*'c')
        TO DO : remove the hardcoded reshape"""

        state_below = state_below.reshape(128,self.size[0],self.size[1],5)
        state_below = state_below.dimshuffle(3,1,2,0)
        return state_below

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):
        return 0

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):
        return 0

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space

        if not isinstance(space, VectorSpace):
            raise TypeError("The input to a Window layer should be a "
                            "VectorSpace,  but layer " + self.layer_name +
                            " got " + str(type(self.input_space)))


        self.output_space = Conv2DSpace(
                                shape=self.size,
                                num_channels=5,
                                axes=self.axes
                                )

    @wraps(Layer.get_params)
    def get_params(self):
        return []

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        return []