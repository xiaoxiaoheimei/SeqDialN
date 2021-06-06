from torch import nn


class EncoderDecoderModel(nn.ModuleList):
    """Convenience wrapper module, wrapping Encoder and Decoder modules.

    Parameters
    ----------
    encoder: nn.Module
    decoder: nn.Module
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        encoder_output = self.encoder(batch)
        decoder_output = self.decoder(encoder_output, batch)
        return decoder_output
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoder.to(*args, **kwargs)
        self.decoder.to(*args, **kwargs)
        return self