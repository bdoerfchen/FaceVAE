import keras

class VAEDescriptor:
    """
    An object describing the structure of a VAE
    """
    def __init__(self) -> None:
        pass

    def createModel(self) -> (keras.models.Model, keras.models.Model):
        """Creates and returns the encoder and decoder.

        Info:
            The created encoder model's output is expected to be (mean, log_variance)

        Args:
            None
        
        Returns:
            encoder (Model): The encoder model
            decoder (Model): The decoder model
        """
        return (None,None)
    
    def getObjectContext(self) -> None:
        pass