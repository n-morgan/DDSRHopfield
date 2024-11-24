
from torchtune.models.mistral import mistral_tokenizer



class Digitize(): 
    '''
    provides methods to encode and decode text
    '''

    tknizer = "/Users/velocity/Documents/Holder/Project/CodingStuff/14VICUNA/llama.cpp/models/Mistral/Transformers/mistral/tokenizer.model"

    m_tokenizer = mistral_tokenizer(tknizer)


    def __init__(self, text, padding=None):
        '''
        text object
        '''
        
        self.padding = padding
        self.text = text
        self.encoding = None
        self.decoding = None
    
    def encode(self, tknizer=tknizer):
        '''

        input:  tokenizer
        output: array of integers corresponding to tokenization
        '''
            

        
        encoding = Digitize.m_tokenizer.encode(self.text)
        
        self.encoding = encoding

        if self.padding is not None:
            pad_amount =  self.padding - len(self.encoding)
            self.encoding = self.encoding + [0]*pad_amount
        return self.encoding



    def decode(self,  encoding, tknizer=tknizer):
        '''
            
        input: tokenizer, encoding
        output: string corresponding to decoded encoding
        '''

        decoding = Digitize.m_tokenizer.decode(encoding)
        self.decoding = decoding



        return self.decoding

    

if __name__ == "__main__":
    digitizer = Digitize("The quick brown fox", padding=512)
    print(len(digitizer.encode()))

    digitizer2 = Digitize("The quick brown", padding=512)
    print(len(digitizer2.encode()))
    sentence = Digitize("the fast cat",100)
    print(sentence.encode())
    print(sentence.padding)
    print(len(sentence.encode()))

    print(len(Digitize.m_tokenizer))

