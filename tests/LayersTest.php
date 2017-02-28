<?php
use PHPUnit\Framework\TestCase;

require_once __DIR__ . '/../index.php';

class LayersTest extends TestCase
{

    public function test_Relu() {
        $relu = new Relu();
        $x = Matrix::createFromData([
            [-2.0, -1.0, 0.0, 1.0, 2.0,]
        ]);
        $out = $relu->forward($x);
        $this->assertEquals([[0.0, 0.0, 0.0, 1.0, 2.0]], $out->row);

        $dout = Matrix::createFromData([
            [1, 2, 3, 4, 5]
        ]);
        $back = $relu->backward($dout);
        $this->assertEquals([[0.0, 0.0, 0.0, 4.0, 5.0]], $back->row);
    }

    public function test_softmax() {
        $m = Matrix::createFromData([
            [1, 2],
            [3, 4],
        ]);
        $expected = [
            [exp(1) / (exp(1) + exp(2)), exp(2) / (exp(1) + exp(2))],
            [exp(3) / (exp(3) + exp(4)), exp(4) / (exp(3) + exp(4))],
        ];
        $this->assertEquals($expected, SoftmaxWithLoss::softmax($m)->row);
    }

    public function test_cross_entropy_error() {
        $y = Matrix::createFromData([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]);
        $t  = Matrix::createFromData([
            [0, 1, 0],
            [1, 0, 0],
        ]);
        $expected = (- log(0.2) - log(0.4)) / 2.0;
        $E = SoftmaxWithLoss::cross_entropy_error($y, $t);
        $this->assertTrue(abs($expected - $E) < 1E-5);

        $x = Matrix::createFromData([
            [0.1, 0.4, 0.5]
        ]);
        $t = Matrix::createFromData([
            [0.0, 0.0, 1.0],
        ]);
        $expected = 0.69314718055994529;
        $E = SoftmaxWithLoss::cross_entropy_error($x, $t);
        $this->assertTrue(abs($expected - $E) < 1E-6);
    }

    public function test_Affine() {
        $W = Matrix::createFromData([
            [1, 2, 3],
            [4, 5, 6],
        ]);
        $b = Matrix::createFromData([[10, 20, 30]]);
        $x = Matrix::createFromData([
            [1, 2],
            [3, 4],
            [5, 6],
        ]);
        $affine = new Affine($W, $b);
        $expected = [
            [19, 32, 45],
            [29, 46, 63],
            [39, 60, 81],
        ];
        $this->assertEquals($expected, $affine->forward($x)->row);
    }

    public function test_SoftmaxWithLoss() {
        $smwl = new SoftmaxWithLoss();
        $x = Matrix::createFromData([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ]);
        $t = Matrix::createFromData([
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        ]);
        $loss = $smwl->forward($x, $t);
        $expected = 9.4586297444267107;
        var_dump($loss);
        $this->assertTrue(abs($expected - $loss) < 1E-6);
    }
}