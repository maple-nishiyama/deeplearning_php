<?php
use PHPUnit\Framework\TestCase;

require_once __DIR__ . '/../index.php';

class MatrixTest extends TestCase
{

    public function test_zerosLike() {
        $m = Matrix::createFromData([
            [1, 2, 3],
            [4, 5, 6],
        ]);
        $zeros = Matrix::zerosLike($m);
        $expected = [
            [0, 0, 0],
            [0, 0, 0],
        ];
        $this->assertEquals($expected, $zeros->row);
    }

    public function test_onesLike() {
        $m = Matrix::createFromData([
            [1, 2, 3],
            [4, 5, 6],
        ]);
        $ones = Matrix::onesLike($m);
        $expected = [
            [1, 1, 1],
            [1, 1, 1],
        ];
        $this->assertEquals($expected, $ones->row);
    }

    public function test_sumCol() {
        $m = Matrix::createFromData([
            [1, 2, 3],
            [4, 5, 6],
        ]);
        $expected = [
            [6],
            [15],
        ];
        $sumCol = $m->sumCol();
        $this->assertEquals($expected, $sumCol->row);
    }

    public function test_sumRow() {
        $m = Matrix::createFromData([
            [1, 2, 3],
            [4, 5, 6],
        ]);
        $expected = [
            [5, 7, 9],
        ];
        $sumRow = $m->sumRow();
        $this->assertEquals($expected, $sumRow->row);
    }

}