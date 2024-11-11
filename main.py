from digitize import Digitize

digitization = Digitize("hello world")


encoding = digitization.encode()
decoding = digitization.decode(encoding)

print(encoding, decoding)
