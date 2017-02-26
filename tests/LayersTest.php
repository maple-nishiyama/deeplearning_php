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
}