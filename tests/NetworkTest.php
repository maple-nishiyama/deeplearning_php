<?php
use PHPUnit\Framework\TestCase;

require_once __DIR__ . '/../index.php';

class NetworkTest extends TestCase
{
    public function test_parameter_update() {
        $network = new TwoLayerNeuralNet(10, 5, 3);
        $originalW1_1_1 = $network->params['W1']->get(1, 1);
        $originalAffineW1_1_1 = $network->layers['Affine1']->W->get(1, 1);
        $this->assertEquals($originalAffineW1_1_1, $originalW1_1_1);
        $newParam = new Matrix(10, 5);
        $newParam->set(1, 1, 100);
        $network->params['W1'] = $newParam;
        $this->assertEquals($network->params['W1']->get(1, 1), 100);
        $this->assertEquals($network->layers['Affine1']->W->get(1, 1), 100);
    }

    public function test_backpropagationGradient() {
        $network = new TwoLayerNeuralNet(3, 5, 4);
        $t = Matrix::createFromData([[1, 0, 0, 0]]);
        $x = Matrix::createFromData([[0.1, 0.2, 0.3]]);
        $grads = $network->backpropagationGradient($x, $t);
        $zeros = Matrix::zerosLike($grads['W1']);
        $this->assertNotEquals($zeros->toArray(), $grads['W1']->toArray());
    }

    public function test_gradientDescent() {
        $learningRate = 0.1;
        $network = new TwoLayerNeuralNet(3, 5, 4);
        $t = Matrix::createFromData([[1, 0, 0, 0]]);
        $x = Matrix::createFromData([[0.1, 0.2, 0.3]]);
        $originalParamW1 = $network->params['W1']->toArray();
        $grad = $network->backpropagationGradient($x, $t);
        $network->params['W1'] = $network->params['W1']->plus($grad['W1']->scale(-$learningRate));
        $result = $network->params['W1']->toArray();
        $this->assertNotEquals($originalParamW1, $result);
    }
}