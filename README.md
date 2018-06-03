# capsnet
Implementation of Hinton's Capsule Network for text (digit) classification. 

This is an extremely interesting archietcture, for it does away with the typically used pooling layers of traditional convolutional neural networks, replacing them with a routing scheme that makes the enncoding of information an _equivariant_ process, not simply a translation invariant one. In principle this should eliminate the need for data augmentation, since pose is inherently captured by the architecture and does not need to be reintroduced by artificial means.

Below is a diagram of a a layer of three capsules routing into a layer of two capsules for simplicity. It should give an idea of what exactly is going on. For the initial layer, the only section of the diagram that matters is the last one, where the input vectors are squashed by the vector-valued squashing function q. Every vertex node of this graph represents a vector.

![CapsDiagram](https://raw.githubusercontent.com/JamesFitzpatrickTP/capsnet/master/CapsNet.PNG)
