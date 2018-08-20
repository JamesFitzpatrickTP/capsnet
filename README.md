# capsnet
Implementation of Lalonde's capsule segmentation network

This is an extremely interesting archietcture, for it does away with the typically used pooling layers of traditional convolutional neural networks, replacing them with a routing scheme that makes the encoding of information an _equivariant_ process, not simply a translation invariant one. In principle this should eliminate the need for data augmentation, since pose is inherently captured by the architecture and does not need to be reintroduced by artificial means.

Below is a diagram of a a layer of three capsules routing into a layer of two capsules for simplicity. It should give an idea of what exactly is going on. For the initial layer, the only section of the diagram that matters is the last one, where the input vectors are squashed by the vector-valued squashing function q. Every vertex node of this graph represents a vector. This is the original scheme
developed for capsules.

![CapsDiagram](https://raw.githubusercontent.com/JamesFitzpatrickTP/capsnet/master/CapsNet.PNG)

Such a scheme, with fully-connected capsule layers, is not feasible for large images. To deal with this, Lalonde proposed locally-restricted capsules, where a single weighting tensor would be used to 'convolve' over an input tensor to produce the new one. By 
sharing the weights the number of parameters needed to produce a model for typical images can be reduced drastically. Normal convolutions
rely on the Hadamard product:

![CapsDiagram](https://raw.githubusercontent.com/JamesFitzpatrickTP/capsnet/master/Hadamard.png)

However, the interpretation of our tensors in such a way that the output of our capsule operations are vectors that encode within their
orientation the appearance of certain features and within their magnitude the probability of an object being present at that location
requires that we generalise the notion of a matrix product to higher-rank objects:

![CapsDiagram](https://raw.githubusercontent.com/JamesFitzpatrickTP/capsnet/master/MatConv.png)
