# Brainy
Neural network implementation in PHP, packaged for Composer.

Based heavily on [the Tremani neural network](https://github.com/infostreams/neural-network) by [Edward Akerboom](https://github.com/infostreams) but stripped down, tidied up and packaged for Composer. 

## Installation
Install Brainy via Composer like this:

```bash
composer require lambdacasserole/brainy
```

Or alternatively, if you're using the PHAR (make sure the `php.exe` executable is in your PATH):

```
php composer.phar require lambdacasserole/brainy
```

## Usage
Create a new neural network instance like this:

```php
// Create a new neural network with 3 input neurons, one layer of 4 hidden neurons, and 1 output neuron.
$network = new NeuralNetwork(3, 4, 1);
```

Add training data to your new network thusly:

```php
// Add training data to the network. In this case, we want the network to learn the 'XOR' function.
$network->addTrainingData([-1, -1, 1], [-1]);
$network->addTrainingData([-1, 1, 1], [1]);
$network->addTrainingData([1, -1, 1], [1]);
$network->addTrainingData([1, 1, 1], [-1]);
```

Then begin training:

```php
// Train in a maximum of 1000 epochs to a maximum error rate of 0.01.
$success = $network->train(1000, 0.01);
```

Now put it to work:

```
$output = $network->calculate([-1, -1, 1]); // Gives [-1].
$output = $network->calculate([-1, 1, 1]); // Gives [1].
```

## Compatibility
Uses new array syntax and splats, so won't work on any PHP versions earlier than 5.6.

## Further Reading
The [original repository](https://github.com/infostreams/neural-network) contains more comprehensive documentation, though it may need adjusting slightly due to modifications made to it in this version.
