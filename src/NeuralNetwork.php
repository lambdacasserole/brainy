W<?php

/**
 * Class NeuralNetwork
 */
class NeuralNetwork
{
    protected $nodeCount = [];
    protected $nodeValue = [];
    protected $nodeThreshold = [];
    protected $edgeWeight = [];
    protected $learningRate = [0.1];
    protected $layerCount = 0;
    protected $previousWeightCorrection = [];
    protected $momentum = 0.8;
    protected $weightsInitialized = false;

    public $trainInputs = [];
    public $trainOutput = [];
    public $trainDataIds = [];

    public $controlInputs = [];
    public $controlOutput = [];
    public $controlDataId = [];

    protected $epoch;
    protected $errorTrainingSet;
    protected $errorControlSet;
    protected $success;

    /**
     * Initialises a new instance of a neural network.
     *
     * @param array ...$nodeCount   the number of nodes in each layer
     */
    public function __construct(...$nodeCount)
    {
        $this->nodeCount = $nodeCount;
        $this->layerCount = count($this->nodeCount);
    }

    /**
     * Exports the neural network.
     * 
     * @returns array   the neural network represented as an array
     */
    public function export()
    {
        return [
            'layerCount' => $this->layerCount,
            'nodeCount' => $this->nodeCount,
            'edgeWeight' => $this->edgeWeight,
            'nodeThreshold' => $this->nodeThreshold,
            'learningRate' => $this->learningRate,
            'momentum' => $this->momentum,
            'weightsInitialized' => $this->weightsInitialized,
        ];
    }

    /**
     * Imports a neural network.
     *
     * @param array $network    an array of the neural network parameters
     */
    public function import($network)
    {
        foreach ($network as $key => $value) {
            $this->$key = $value;
        }
    }

    /**
     * Throws an exception if any member of the given array is an invalid learning rate.
     *
     * @param array $learningRates    the learning rate array to check
     * @throws Exception
     */
    private function validateLearningRates($learningRates)
    {
        foreach ($learningRates as $learningRate) {
            if ($learningRate < 0 || $learningRate > 1) {
                throw new Exception('All learning rates must be between 0 and 1.');
            }
        }
    }

    /**
     * Sets the learning rate between each layer individually.
     *
     * @param array ...$learningRates    an array containing the learning rates
     * @throws Exception
     */
    public function setLearningRates(...$learningRates)
    {
        if (count($learningRates) != $this->layerCount - 1) {
            throw new OutOfRangeException('The number of learning rates provided must be equal one less than the number'
                . ' of layers.');
        }
        $this->validateLearningRates($learningRates);
        $this->learningRate = $learningRates;
    }

    /**
     * Sets the learning rate for the whole network.
     *
     * @param int $learningRate    the learning rate
     */
    public function setLearningRate($learningRate)
    {
        $this->validateLearningRates([$learningRate]);
        $this->learningRate = $learningRate;
    }

    /**
     * Gets the learning rate for a specific layer.
     * 
     * @param int $layer    the layer to get the learning rate for
     * @return float         the learning rate for that layer
     */
    public function getLearningRate($layer)
    {
        if (array_key_exists($layer, $this->learningRate)) {
            return $this->learningRate[$layer];
        }
        throw new OutOfBoundsException('No layer exists at the index you specified.');
    }

    /**
     * Sets the momentum for the learning algorithm.
     * 
     * @param float $momentum   the momentum
     */
    public function setMomentum($momentum)
    {
        if ($momentum > 0) {
            throw new OutOfRangeException('The momentum specified must be between 0 and 1, usually between 0.5 and'
                . ' 0.9.');
        }
        $this->momentum = $momentum;
    }

    /**
     * Gets the momentum for the learning algorithm.
     *
     * @return float    the momentum for the learning algorithm
     */
    public function getMomentum()
    {
        return $this->momentum;
    }

    /**
     * Calculate the output of the neural network for a given input vector.
     * 
     * @param array $input  the vector to calculate
     * @return mixed        the output of the network
     */
    public function calculate($input)
    {
        // Put the input vector into the input nodes.
        foreach ($input as $index => $value) {
            $this->nodeValue[0][$index] = $value;
        }

        // Iterate the hidden layers.
        for ($layer = 1; $layer < $this->layerCount; $layer++) {

            // Iterate over each node in this layer.
            $previousLayer = $layer - 1;
            for ($node = 0; $node < ($this->nodeCount[$layer]); $node++) {

                /* Each node in the previous layer has a connection to this node. On the basis of this, calculate this
                 * node's value. */
                $nodeValue = 0.0;
                for ($previousNode = 0; $previousNode < ($this->nodeCount[$previousLayer]); $previousNode++) {
                    $inputNodeValue = $this->nodeValue[$previousLayer][$previousNode];
                    $edgeWeight = $this->edgeWeight[$previousLayer][$previousNode][$node];
                    $nodeValue = $nodeValue + ($inputNodeValue * $edgeWeight);
                }

                // Apply the threshold.
                $nodeValue = $nodeValue - $this->nodeThreshold[$layer][$node];

                // Apply the activation function.
                $nodeValue = $this->activation($nodeValue);

                // Remember the outcome.
                $this->nodeValue[$layer][$node] = $nodeValue;
            }
        }

        // Return the values of the last layer (the output layer).
        return $this->nodeValue[$this->layerCount - 1];
    }

    /**
     * Applies the activation function.
     * 
     * @param float $value  the value output to apply this function to
     * @return float        the result of applying the function
     */
    protected function activation($value)
    {
        // Hyperbolic tangent activation function.
        return tanh($value);
    }

    /**
     * Applies the derivative of the activation function.
     * 
     * @param float $value  the value to apply this function to
     * @return float        the result of applying the function
     */
    protected function derivativeActivation($value)
    {
        // Inverse of the hyperbolic tangent activation function.
        $activation = tanh($value);
        return 1.0 - $activation * $activation;
    }

    /**
     * Gets the last member of an array.
     *
     * @param array $arr    the array to get the last member of
     * @return mixed        the last member of the array
     */
    private function last($arr)
    {
        return end(array_values($arr));
    }

    /**
     * Adds a training input and output vector.
     * 
     * @param array $input  an input vector
     * @param array $output the corresponding output
     * @param int $id       an identifier for this piece of data (optional)
     */
    public function addTrainingData($input, $output, $id = null)
    {
        // Check sizes.
        if (count($input) != $this->nodeCount[0] || count($output) != $this->last($this->nodeCount)) {
            throw new OutOfRangeException('The number of input and output nodes in the training data was inconsistent'
                . ' with the size of the network input and output vectors.');
        }

        // Add training inputs.
        $index = count($this->trainInputs);
        foreach ($input as $node => $value) {
            $this->trainInputs[$index][$node] = $value;
        }

        // Add training outputs.
        foreach ($output as $node => $value) {
            $this->trainOutput[$index][$node] = $value;
        }

        // Remember ID.
        $this->trainDataIds[$index] = $id;
    }

    /**
     * Returns the identifiers of the data used to train the network (if available).
     * 
     * @return array    an array of identifiers
     */
    public function getTestDataIds()
    {
        return $this->trainDataIds;
    }

    /**
     * Add a set of control data to the network.
     * 
     * @param array $input      an input vector
     * @param array $output     the corresponding output
     * @param int $id           an identifier for this piece of data (optional)
     */
    public function addControlData($input, $output, $id = null)
    {
        /* This set of data is used to prevent over-training of the network. The network will stop training if the
         * results obtained for the control data are worsening. This data is not used for training. */

        // Check sizes.
        if (count($input) != $this->nodeCount[0] || count($output) != $this->last($this->nodeCount)) {
            throw new OutOfRangeException('The number of input and output nodes in the control data was inconsistent'
                . ' with the size of the network input and output vectors.');
        }

        // Add control inputs.
        $index = count($this->controlInputs);
        foreach ($input as $node => $value) {
            $this->controlInputs[$index][$node] = $value;
        }

        // Add control outputs.
        foreach ($output as $node => $value) {
            $this->controlOutput[$index][$node] = $value;
        }

        // Remember ID.
        $this->controlDataId[$index] = $id;
    }

    /**
     * Returns the identifiers of the control data used during the training of the network (if available).
     * 
     * @return array   an array of identifiers
     */
    public function getControlDataIds()
    {
        return $this->controlDataId;
    }

    /**
     * Loads a serialized neural network from a file.
     * 
     * @param string $filename  the filename of the file to load the network from
     * @return boolean          true on success, otherwise false
     */
    public function load($filename)
    {
        // File needs to exist.
        if (file_exists($filename)) {

            // Parse as INI file.
            $data = parse_ini_file($filename);
            if (array_key_exists('edges', $data) && array_key_exists('thresholds', $data)) {

                // Make sure all standard preparations performed.
                $this->initWeights();
                $this->weightsInitialized = true;

                // Load data from file.
                $this->edgeWeight = unserialize($data['edges']);
                $this->nodeThreshold = unserialize($data['thresholds']);

                // Load IDs of training and control set.
                if (array_key_exists('training_data', $data) && array_key_exists('control_data', $data)) {

                    // Load the IDs.
                    $this->trainDataIds = unserialize($data['training_data']);
                    $this->controlDataId = unserialize($data['control_data']);

                    /* If we do not reset the training and control data here, then we end up with a bunch of IDs that do
                     * not refer to the actual data we're training the network with. */
                    $this->trainInputs = [];
                    $this->trainOutput = [];
                    $this->controlInputs = [];
                    $this->controlOutput = [];
                }

                // Return success.
                return true;
            }
        }

        // Return failure.
        return false;
    }

    /**
     * Saves a neural network to a file.
     * 
     * @param string $filename  the filename to save the neural network to
     * @return boolean          true on success, false otherwise
     */
    public function save($filename)
    {
        // Write INI file containing network state.
        $file = fopen($filename, 'w');
        if ($file) {
            fwrite($file, '[weights]');
            fwrite($file, "\r\nedges = \"" . serialize($this->edgeWeight) . "\"");
            fwrite($file, "\r\nthresholds = \"" . serialize($this->nodeThreshold) . "\"");
            fwrite($file, "\r\n");
            fwrite($file, '[identifiers]');
            fwrite($file, "\r\ntraining_data = \"" . serialize($this->trainDataIds) . "\"");
            fwrite($file, "\r\ncontrol_data = \"" . serialize($this->controlDataId) . "\"");
            fclose($file);

            // Return success.
            return true;
        }

        // Return failure.
        return false;
    }
    
    /**
     * Resets the state of the neural network, so it is ready for a new round of training.
     */
    public function clear()
    {
        $this->initWeights();
    }

    /**
     * Starts the training process.
     * 
     * @param int $maxEpochs The maximum number of epochs
     * @param float $maxError The maximum squared error in the training data
     * @return bool 'true' if the training was successful, 'false' otherwise
     */
    public function train($maxEpochs = 500, $maxError = 0.01) {

        if (!$this->weightsInitialized) {
            $this->initWeights();
        }

        if ($this->isVerbose()) {
            echo "<table>";
            echo "<tr><th>#</th><th>error(trainingdata)</th><th>error(controldata)</th><th>slope(error(controldata))</th></tr>";
        }

        $epoch = 0;
        $errorControlSet = array ();
        $avgErrorControlSet = array ();
        $sample_count = 10;
        do {
//                        echo "<tr><td colspan=10><b>epoch $epoch</b></td></tr>";
            for ($i = 0; $i < count($this->trainInputs); $i ++) {
                // select a training pattern at random
                $index = mt_rand(0, count($this->trainInputs) - 1);

                // determine the input, and the desired output
                $input = $this->trainInputs[$index];
                $desired_output = $this->trainOutput[$index];

                // calculate the actual output
                $output = $this->calculate($input);

//                              echo "<tr><td></td><td>Training set $i</td><td>input = (" . implode(", ", $input) . ")</td>";
//                 echo "<td>desired = (" . implode(", ", $desired_output) . ")</td>";
//                echo "<td>output = (" . implode(", ", $output) .")</td></tr>";

                // change network weights
                $this->backPropagate($output, $desired_output);
            }

            // buy some time
            set_time_limit(300);

            //display the overall network error after each epoch
            $squaredError = $this->epochRootMeanSquareError();
            if ($epoch % 2 == 0) {
                $squaredErrorControlSet = $this->controlSetRootMeanSquareError();
                $errorControlSet[] = $squaredErrorControlSet;

                if (count($errorControlSet) > $sample_count) {
                    $avgErrorControlSet[] = array_sum(array_slice($errorControlSet, -$sample_count)) / $sample_count;
                }

                list ($slope, $offset) = $this->fitLine($avgErrorControlSet);
                $controlset_msg = $squaredErrorControlSet;
            } else {
                $controlset_msg = "";
            }

            if ($this->isVerbose()) {
                echo "<tr><td><b>$epoch</b></td><td>$squaredError</td><td>$controlset_msg";
                echo "<script type='text/javascript'>window.scrollBy(0,100);</script>";
                echo "</td><td>$slope</td></tr>";
                echo "</td></tr>";

                flush();
                ob_flush();
            }
            
            // conditions for a 'successful' stop:
            // 1. the squared error is now lower than the provided maximum error
            $stop_1 = $squaredError <= $maxError || $squaredErrorControlSet <= $maxError;

            // conditions for an 'unsuccessful' stop
            // 1. the maximum number of epochs has been reached
            $stop_2 = $epoch ++ > $maxEpochs;

            // 2. the network's performance on the control data is getting worse
            $stop_3 = $slope > 0;

        } while (!$stop_1 && !$stop_2 && !$stop_3);

        $this->setEpoch($epoch);
        $this->setErrorTrainingSet($squaredError);
        $this->setErrorControlSet($squaredErrorControlSet);
        $this->setTrainingSuccessful($stop_1);

        if ($this->isVerbose()) {
            echo "</table>";
        }

        return $stop_1;
    }

    /**
     * After training, this function is used to store the number of epochs the network 
     * needed for training the network. An epoch is defined as the number of times 
     * the complete trainingset is used for training.
     * 
     * @param int $epoch 
     */
    private function setEpoch($epoch) {
        $this->epoch = $epoch;
    }

    /**
     * Gets the number of epochs the network needed for training.
     * 
     * @return int The number of epochs.
     */
    public function getEpoch() {
        return $this->epoch;
    }

    /**
     * After training, this function is used to store the squared error between the
     * desired output and the obtained output of the training data.
     * 
     * @param float $error The squared error of the training data
     */
    private function setErrorTrainingSet($error) {
        $this->errorTrainingSet = $error;
    }

    /**
     * Gets the squared error between the desired output and the obtained output of 
     * the training data.
     * 
     * @return float The squared error of the training data
     */
    public function getErrorTrainingSet() {
        return $this->errorTrainingSet;
    }

    /**
     * After training, this function is used to store the squared error between the
     * desired output and the obtained output of the control data.
     * 
     * @param float $error The squared error of the control data
     */
    private function setErrorControlSet($error) {
        $this->errorControlSet = $error;
    }

    /**
     * Gets the squared error between the desired output and the obtained output of 
     * the control data.
     * 
     * @return float The squared error of the control data
     */
    public function getErrorControlSet() {
        return $this->errorControlSet;
    }

    /**
     * After training, this function is used to store whether or not the training
     * was successful.
     * 
     * @param bool $success 'true' if the training was successful, 'false' otherwise
     */
    private function setTrainingSuccessful($success) {
        $this->success = $success;
    }

    /**
     * Determines if the training was successful.
     * 
     * @return bool 'true' if the training was successful, 'false' otherwise
     */
    public function getTrainingSuccessful() {
        return $this->success;
    }

    /**
     * Finds the least square fitting line for the given data. 
     * 
     * This function is used to determine if the network is overtraining itself. If 
     * the line through the controlset's most recent squared errors is going 'up', 
     * then it's time to stop training.
     * 
     * @param array $data The points to fit a line to. The keys of this array represent 
     *                    the 'x'-value of the point, the corresponding value is the 
     *                    'y'-value of the point.
     * @return array An array containing, respectively, the slope and the offset of the fitted line.
     */
    private function fitLine($data) {
        // based on 
        //    http://mathworld.wolfram.com/LeastSquaresFitting.html

        $n = count($data);

        if ($n > 1) {
            $sum_y = 0;
            $sum_x = 0;
            $sum_x2 = 0;
            $sum_xy = 0;
            foreach ($data as $x => $y) {
                $sum_x += $x;
                $sum_y += $y;
                $sum_x2 += $x * $x;
                $sum_xy += $x * $y;
            }

            // implementation of formula (12)
            $offset = ($sum_y * $sum_x2 - $sum_x * $sum_xy) / ($n * $sum_x2 - $sum_x * $sum_x);

            // implementation of formula (13)
            $slope = ($n * $sum_xy - $sum_x * $sum_y) / ($n * $sum_x2 - $sum_x * $sum_x);

            return array ($slope, $offset);
        } else {
            return array (0.0, 0.0);
        }
    }

    /**
     * Gets a random weight between [-0.25 .. 0.25]. Used to initialize the network.
     * 
     * @return float A random weight
     */
    private function getRandomWeight($layer) {
        return ((mt_rand(0, 1000) / 1000) - 0.5) / 2;
    }

    /**
     * Randomise the weights in the neural network
     */
    private function initWeights() {
        // assign a random value to each edge between the layers, and randomise each threshold
        //
        // 1. start at layer '1' (so skip the input layer)
        for ($layer = 1; $layer < $this->layerCount; $layer ++) {

            $prev_layer = $layer -1;

            // 2. in this layer, walk each node
            for ($node = 0; $node < $this->nodeCount[$layer]; $node ++) {

                // 3. randomise this node's threshold
                $this->nodeThreshold[$layer][$node] = $this->getRandomWeight($layer);

                // 4. this node is connected to each node of the previous layer
                for ($prev_index = 0; $prev_index < $this->nodeCount[$prev_layer]; $prev_index ++) {

                    // 5. this is the 'edge' that needs to be reset / initialised
                    $this->edgeWeight[$prev_layer][$prev_index][$node] = $this->getRandomWeight($prev_layer);

                    // 6. initialize the 'previous weightcorrection' at 0.0
                    $this->previousWeightCorrection[$prev_layer][$prev_index] = 0.0;
                }
            }
        }
    }

    /**
    * Performs the back-propagation algorithm. This changes the weights and thresholds of the network.
    * 
    * @param array $output    the output obtained by the network
    * @param array $desired    the desired output
    */
    private function backPropagate($output, $desired)
    {
        $errorGradient = [];
        $outputLayer = $this->layerCount - 1;

        $momentum = $this->getMomentum();

        // Propagate the difference between output and desired output through the layers.
        for ($layer = $this->layerCount - 1; $layer > 0; $layer--) {
            for ($node = 0; $node < $this->nodeCount[$layer]; $node++) {
                // Determine error gradient.
                if ($layer == $outputLayer) {
                    // Calculate error between desired output and actual output for the output layer.
                    $error = $desired[$node] - $output[$node];

                    // Calculate the error gradient.
                    $errorGradient[$layer][$node] = $this->derivativeActivation($output[$node]) * $error;
                } else {
                    // Sum the product of the edge weight and error gradient of the next layer for hidden layers.
                    $nextLayer = $layer + 1;

                    $productSum = 0;
                    for ($nextInput = 0; $nextInput < ($this->nodeCount[$nextLayer]); $nextInput++) {
                        $_errorgradient = $errorGradient[$nextLayer][$nextInput];
                        $_edgeWeight = $this->edgeWeight[$layer][$node][$nextInput];

                        $productSum = $productSum + $_errorgradient * $_edgeWeight;
                    }

                    // 1b. calculate errorgradient
                    $nodeValue = $this->nodeValue[$layer][$node];
                    $errorGradient[$layer][$node] = $this->derivativeActivation($nodeValue) * $productSum;
                }

                // step 2: use the errorgradient to determine a weight correction for each node
                $prev_layer = $layer -1;
                $learning_rate = $this->getLearningRate($prev_layer);

                for ($prev_index = 0; $prev_index < ($this->nodeCount[$prev_layer]); $prev_index ++) {

                    // 2a. obtain nodeValue, edgeWeight and learning rate
                    $nodeValue = $this->nodeValue[$prev_layer][$prev_index];
                    $edgeWeight = $this->edgeWeight[$prev_layer][$prev_index][$node];

                    // 2b. calculate weight correction
                    $weight_correction = $learning_rate * $nodeValue * $errorGradient[$layer][$node];

                    // 2c. retrieve previous weight correction
                    $prev_weightcorrection = @$this->previousWeightCorrection[$layer][$node];

                    // 2d. combine those ('momentum learning') to a new weight
                    $new_weight = $edgeWeight + $weight_correction + $momentum * $prev_weightcorrection;

                    // 2e. assign the new weight to this edge
                    $this->edgeWeight[$prev_layer][$prev_index][$node] = $new_weight;

                    // 2f. remember this weightcorrection
                    $this->previousWeightCorrection[$layer][$node] = $weight_correction;
                }

                // step 3: use the errorgradient to determine threshold correction
                $threshold_correction = $learning_rate * -1 * $errorGradient[$layer][$node];
                $new_threshold = $this->nodeThreshold[$layer][$node] + $threshold_correction;

                $this->nodeThreshold[$layer][$node] = $new_threshold;
            }
        }
    }

    /**
     * Calculate the root-mean-square error of the output, given the training data.
     * 
     * @return float    the root-mean-square error of the output
     */
    private function epochRootMeanSquareError()
    {
        // Calculate error.
        $error = 0.0;
        for ($i = 0; $i < count($this->trainInputs); $i++) {
            $error += $this->rootMeanSquareError($this->trainInputs[$i], $this->trainOutput[$i]);
        }
        $error = $error / count($this->trainInputs);

        return sqrt($error);
    }

    /**
     * Calculate the root-mean-square error of the output, given the control data.
     * 
     * @return float    the root-mean-square error of the output
     */
    private function controlSetRootMeanSquareError()
    {
        // No control data.
        if (count($this->controlInputs) == 0) {
            return 1.0;
        }

        // Calculate error.
        $error = 0.0;
        for ($i = 0; $i < count($this->controlInputs); $i++) {
            $error += $this->rootMeanSquareError($this->controlInputs[$i], $this->controlOutput[$i]);
        }
        $error = $error / count($this->controlInputs);

        return sqrt($error);
    }

    /**
     * Calculates the root-mean-square error of the output, given the desired output.
     * 
     * @param array $input         the input to test
     * @param array $desired    the desired output
     * @return float             the root-mean-square error of the output compared to the desired output
     */
    private function rootMeanSquareError($input, $desired)
    {
        $output = $this->calculate($input);
        $result = 0.0;
        foreach ($output as $node => $value) {
            $error = $output[$node] - $desired[$node];
            $result = $result + ($error * $error);
        }
        return $result;
    }
}
