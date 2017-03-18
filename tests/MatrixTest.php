<?php
use PHPUnit\Framework\TestCase;

//require_once __DIR__ . '/../index.php';

class MatrixTest extends TestCase
{

    public function test_construct() {
        $m = new Matrix(3, 4);
        $expected = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ];
        $this->assertEquals($expected, $m->toArray());
    }

    public function test_createFromData() {
        $data = [
            [1, 2, 3],
            [4, 5, 6]
        ];
        $m = Matrix::createFromData($data);
        $this->assertEquals($data, $m->toArray());
    }

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
        $this->assertEquals($expected, $zeros->toArray());
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
        $this->assertEquals($expected, $ones->toArray());
    }

    public function test_shape() {
        $m = new Matrix(3, 4);
        $this->assertEquals([3, 4], $m->shape());
        $m = new Matrix(1, 5);
        $this->assertEquals([1, 5], $m->shape());
        $m = new Matrix(4, 1);
        $this->assertEquals([4, 1], $m->shape());
    }

    public function test_mul() {
        $m1 = Matrix::createFromData([[1, 2, 3], [4, 5, 6]]);
        $m2 = Matrix::createFromData([[2, 3], [3, 4], [5, 6]]);
        $expected = [
            [1 * 2 + 2 * 3 + 3 * 5, 1 * 3 + 2 * 4 + 3 * 6],
            [4 * 2 + 5 * 3 + 6 * 5, 4 * 3 + 5 * 4 + 6 * 6],
        ];
        $this->assertEquals($expected, $m1->mul($m2)->toArray());
    }

    public function test_componentwise_prod() {
        $m1 = Matrix::createFromData([[1, 2, 3], [4, 5, 6]]);
        $m2 = Matrix::createFromData([[2, 3, 4], [5, 6, 7]]);
        $expected = [
            [1 * 2, 2 * 3, 3 * 4],
            [4 * 5, 5 * 6, 6 * 7],
        ];
        $this->assertEquals($expected, $m1->componentwiseProd($m2)->toArray());
    }

    public function test_plus() {
        $m1 = Matrix::createFromData([[1, 2, 3], [4, 5, 6]]);
        $m2 = Matrix::createFromData([[2, 3, 4], [5, 6, 7]]);
        $expected = [
            [1 + 2, 2 + 3, 3 + 4],
            [4 + 5, 5 + 6, 6 + 7],
        ];
        $this->assertEquals($expected, $m1->plus($m2)->toArray());

        $b = Matrix::createFromData([[10, 11, 12]]);

        $expected = [
            [1 + 10, 2 + 11, 3 + 12],
            [4 + 10, 5 + 11, 6 + 12],
        ];
        $this->assertEquals($expected, $m1->plus($b)->toArray());
    }

    public function test_scale() {
        $m1 = Matrix::createFromData([[1, 2, 3], [4, 5, 6]]);
        $scale = 5.0;
        $expected = [
            [5, 10, 15],
            [20, 25, 30],
        ];
        $this->assertEquals($expected, $m1->scale($scale)->toArray());
    }

    public function test_minus() {
        $m1 = Matrix::createFromData([[1, 2, 3], [4, 5, 6]]);
        $m2 = Matrix::createFromData([[2, 3, 4], [5, 6, 7]]);
        $expected = [
            [-1, -1, -1],
            [-1, -1, -1],
        ];
        $this->assertEquals($expected, $m1->minus($m2)->toArray());
    }

    public function test_transpose() {
        $m1 = Matrix::createFromData([[1, 2, 3], [4, 5, 6]]);
        $expected = [
            [1, 4],
            [2, 5],
            [3, 6],
        ];
        $this->assertEquals($expected, $m1->transpose()->toArray());
    }

    public function test_argmax() {
        $m1 = Matrix::createFromData([[1, 2, 3], [4, 5, 6]]);
        $expected = [
            [1, 1, 1]
        ];
        $this->assertEquals($expected, $m1->argmax(0)->toArray());

        $expected = [
            [2],
            [2],
        ];
        $this->assertEquals($expected, $m1->argmax(1)->toArray());
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
        $this->assertEquals($expected, $sumCol->toArray());
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
        $this->assertEquals($expected, $sumRow->toArray());
    }

}