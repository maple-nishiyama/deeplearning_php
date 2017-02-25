<?php
ini_set('memory_limit', '1G');

define('DS', DIRECTORY_SEPARATOR);

class Util {

    const MNIST_TRAIN_IMAGE_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz';
    const MNIST_TRAIN_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz';
    const MNIST_TEST_IMAGE_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz';
    const MNIST_TEST_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz';

    const MNIST_DATA_DIR = 'mnist';
    const MNIST_TRAIN_IMAGE_FILE = 'mnist_train_image';
    const MNIST_TRAIN_LABEL_FILE = 'mnist_train_label';
    const MNIST_TEST_IMAGE_FILE = 'mnist_test_image';
    const MNIST_TEST_LABEL_FILE = 'mnist_train_image';

    private static $__mnist_train_image_cache = null;
    private static $__mnist_train_label_cache = null;
    private static $__mnist_test_image_cache = null;
    private static $__mnist_test_label_cache = null;

    static function downloadMnist() {

        $urls = [
            self::MNIST_TRAIN_IMAGE_URL,
            self::MNIST_TRAIN_LABEL_URL,
            self::MNIST_TEST_IMAGE_URL,
            self::MNIST_TEST_LABEL_URL,
        ];

        $files = [
            self::MNIST_TRAIN_IMAGE_FILE,
            self::MNIST_TRAIN_LABEL_FILE,
            self::MNIST_TEST_IMAGE_FILE,
            self::MNIST_TEST_LABEL_FILE,
        ];

        if (!file_exists(self::MNIST_DATA_DIR)) {
            mkdir(self::MNIST_DATA_DIR);
        }

        for ($i = 0; $i < count($urls); $i++) {
            $filePath = self::MNIST_DATA_DIR . DS . $files[$i];
            if (file_exists($filePath)) {
                continue;
            }
            $download = file_get_contents($urls[$i]);
            file_put_contents($filePath.'.gz', $download);
            system("gunzip {$filePath}.gz");
        }
    }

    /**
     * 全学習画像をメモリにロードする
     */
    static function loadData() {
        self::$__mnist_train_image_cache = file_get_contents(self::MNIST_DATA_DIR . DS . self::MNIST_TRAIN_IMAGE_FILE);
        self::$__mnist_train_label_cache = file_get_contents(self::MNIST_DATA_DIR . DS . self::MNIST_TRAIN_LABEL_FILE);
        self::$__mnist_test_image_cache = file_get_contents(self::MNIST_DATA_DIR . DS . self::MNIST_TEST_IMAGE_FILE);
        self::$__mnist_test_label_cache = file_get_contents(self::MNIST_DATA_DIR . DS . self::MNIST_TEST_LABEL_FILE);
    }

    static function getImageAt($image, $index) {
        // ヘッダー解析
        $header = unpack('N4', $image);
        $magic = $header[1];
        $numImages = $header[2];
        $numRow = $header[3];
        $numCol = $header[4];

        $pixels = $numRow * $numCol;
        $start = 4 * 4 + $pixels * $index;
        return array_values(unpack("C{$pixels}", substr($image, $start, $pixels)));
    }

    static function getLabelAt($label, $index, $onehot = true) {
        $header = unpack('N2', $label);
        $magic = $header[1];
        $numLabels = $header[2];
        $label = unpack('C', substr($label, $index + 2, 1));
        if ($onehot) {
            $t = array_fill(0, 10, 0.0);
            $t[$label[1]] = 1.0;
            return $t;
        } else {
            return $label[1];
        }
    }

    static function getTrainPair($index) {
        $data = self::getImageAt(self::$__mnist_train_image_cache, $index);
        $label = self::getLabelAt(self::$__mnist_train_label_cache, $index);
        return [$data, $label];
    }

    static function getTestPair($index) {
        $data = self::getImageAt(self::$__mnist_test_image_cache, $index);
        $label = self::getLabelAt(self::$__mnist_test_label_cache, $index);
        return [$data, $label];
    }

    static function getTrainBatch($batchMask) {
        $data = array_map(function ($rnd) {
            return self::getImageAt(self::$__mnist_train_image_cache, $rnd);
        }, $batchMask);
        $labels = array_map(function($rnd) {
            return self::getLabelAt(self::$__mnist_train_label_cache, $rnd);
        }, $batchMask);

        return [self::toMatrix($data), self::toMatrix($labels)];
   }

    static function getTestBatch($batchMask) {
        $data = array_map(function ($rnd) {
            return self::getImageAt(self::$__mnist_test_image_cache, $rnd);
        }, $batchMask);
        $labels = array_map(function($rnd) {
            return self::getLabelAt(self::$__mnist_test_label_cache, $rnd);
        }, $batchMask);
        return [self::toMatrix($data), self::toMatrix($labels)];
    }

    static function toMatrix($data) {
        $m = new Matrix(0, 0);
        $m->row = $data;
        return $m;
    }

    static function drawImage($pixels, $width, $height) {
        $img = imagecreate($width, $height);
        $black = imagecolorallocate($img, 0, 0, 0);
        imagefill($img, $width, $height, $black);
        for ($i = 0; $i < $height; $i++) {
            for ($j = 0; $j < $width; $j++) {
                $pixel = $pixels[$width * $i + $j];
                $color = imagecolorexact ($img, $pixel, $pixel, $pixel);
                if ($color === -1) {
                    $color = imagecolorallocate($img, $pixel, $pixel, $pixel);
                }
                imagesetpixel($img, $j, $i, $color);
            }
        }
        imagepng($img);
    }

    static function outputTrainImage($index) {
        header("Content-type: image/png");
        $pixels = self::getImageAt(self::$__mnist_train_image_cache, $index);
        self::drawImage($pixels, 28, 28);
    }

    /**
     * 0 から $max までの乱数を $num 個選んで配列にして返す
     *
     * @param int $max
     * @param int $num
     */
    static function randoms($max, $num) {
        $result = array_fill(0, $num, 0);
        for ($i = 0; $i < $num; $i++) {
            $result[$i] = mt_rand(0, $max - 1);
        }
        return $result;
    }
}

class Matrix {

    public $row = null;

    public function __construct($numRow, $numCol) {
        $this->row = [];
        for ($i = 0; $i < $numRow; $i++) {
            $this->row[$i] = array_fill(0, $numCol, 0);
        }
    }
    public function shape() {
        return [count($this->row), count($this->row[0])];
    }

    public function mul(Matrix $m) {
        list($r1, $c1) = $this->shape();
        list($r2, $c2) = $m->shape();
        if($c1 !== $r2) {
            throw new InvalidArgumentException("行列の型が違う");
        }
        $result = new Matrix($r1, $c2);
        for ($i = 0; $i < $r1; $i++) {
            for ($j = 0; $j < $c2; $j++) {
                for ($k = 0; $k < $c1; $k++) {
                    $result->row[$i][$j] += $this->row[$i][$k] * $m->row[$k][$j];
                }
            }
        }
        return $result;
    }

    public function plus(Matrix $m) {
        $s1 = $this->shape();
        $s2 = $m->shape();
        if ($s1 != $s2) {
            if ($s2[0] === 1) {
                // ブロードキャストを試みる
                $m = new Matrix($s1[0], $s1[1]);
                for ($i = 0; $i < $s1[0]; $i++) {
                    $m->row[$i] = $m->row[0];
                }
            } else {
                throw new InvalidArgumentException("行列の型が違う");
            }
        }
        $r = $s1[0];
        $c = $s1[1];
        $result = new Matrix($r, $c);
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $result->row[$i][$j] = $this->row[$i][$j] + $m->row[$i][$j];
            }
        }
        return $result;
    }

    public function scale($a) {
        list($r, $c) = $this->shape();
        $result = new Matrix($r, $c);
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $result->row[$i][$j] =  $a * $this->row[$i][$j];
            }
        }
        return $result;
    }

    public function minus(Matrix $m) {
        return $this->plus($m->scale(-1));
    }

    public function transpose() {
        list($r, $c) = $this->shape();
        $result = new Matrix($c, $r);
        for ($i = 0; $i < $c; $i++) {
            for ($j = 0; $j < $r; $j++) {
                $result->row[$i][$j] = $this->row[$j][$i];
            }
        }
        return $result;
    }

    public function randomize() {
        list($r, $c) = $this->shape();
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $this->row[$i][$j] = mt_rand(0, 1E4) / 1E6;
            }
        }
        return $this;
    }

    public static function argmax(Matrix $m, $dir = 0) {
        list($r, $c) = $m->shape();
        if ($dir == 0) {
            $result = new Matrix(1, $c);
            $tr = $m->transpose();
            for ($i = 0; $i < $c; $i++) {
                $max = max($tr->row[$i]);
                $result->row[0][$i] = array_keys($this->row[$i], $max)[0];
            }
        } else {
            $result = new Matrix($r, 1);
            for ($i = 0; $i < $r; $i++) {
                $max = max($m->row[$i]);
                $result->row[$i][0] = array_keys($this->row[$i], $max)[0];
            }
        }
        return $result;
    }
}

class TwoLayerNeuralNet {

    private $params = [];

    public function __construct($inputSize, $hiddenSize, $outputSize, $weightInitStd=0.01) {
        $this->params = [];
        $this->params['W1'] = (new Matrix($inputSize, $hiddenSize))->randomize();
        $this->params['b1'] = new Matrix(1, $hiddenSize);
        $this->params['W2'] = (new Matrix($hiddenSize, $outputSize))->randomize();
        $this->params['b2'] = new Matrix(1, $outputSize);
    }

    public function predict(Matrix $x) {
        static $i = 0;
        echo "predict $i begin.\n";
        $W1 = $this->params['W1'];
        $W2 = $this->params['W2'];
        $b1 = $this->params['b1'];
        $b2 = $this->params['b2'];

        $a1 = $x->mul($W1)->plus($b1);
        $z1 = self::_sigmoid($a1);
        unset($a1);
        $a2 = $z1->mul($W2)->plus($b2);
        unset($z1);
        $y = self::_softmax($a2);
        echo "predict $i finished.\n";
        $i++;
        return $y;
    }

    public function loss(Matrix $x, Matrix $t) {
        $y = $this->predict($x);
        return self::_cross_entropy_error($y, $t);
    }

    public function accuracy(Matrix $x, Matrix $t) {
        $y = $this->predict($x);
        $ymax = Matrix::argmax($y, $dir=1);
        $tmax = Matrix::argmaax($t, $dir=1);
        $r = $y->shape()[0];
        $acc = 0;
        for ($i = 0; $i < $r; $i++) {
            if ($ymax[$i] === $tmax[$i]) {
                $acc += 1;
            }
        }
        return $acc / $r;
    }

    public function numericalGradient(Matrix $x, Matrix $t) {
        $lossW = function($W) use ($x, $t) { return $this->loss($x, $t); };
        $grads = [];
        echo "numericalGradient for W1\n";
        $grads['W1'] = self::numericalGradientF($lossW, $this->params['W1']);
        echo "numericalGradient for b1\n";
        $grads['b1'] = self::numericalGradientF($lossW, $this->params['b1']);
        echo "numericalGradient for W2\n";
        $grads['W2'] = self::numericalGradientF($lossW, $this->params['W2']);
        echo "numericalGradient for b2\n";
        $grads['b2'] = self::numericalGradientF($lossW, $this->params['b2']);

        return $grads;
    }

    public static function numericalGradientF(Callable $f, Matrix $x) {
        $h = 1E-4;
        list($r, $c) = $x->shape();
        $grad = new Matrix($r, $c);

        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $tmpVal = $x->row[$i][$j];

                $x->row[$i][$j] = $tmpVal + $h;
                $fxh1 = $f($x);

                $x->row[$i][$j] = $tmpVal - $h;
                $fxh2 = $f($x);

                $grad->row[$i][$j] = ($fxh1 - $fxh2) / (2 * $h);
                $x->row[$i][$j] = $tmpVal;
            }
        }

        return $grad;
    }

    private static function _sigmoid(Matrix $m) {
        list($r, $c) = $m->shape();
        $result = new Matrix($r, $c);
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $result->row[$i][$j] = 1 / (1 + exp($m->row[$i][$j]));
            }
        }
        return $result;
    }

    private static function _softmax(Matrix $m) {
        list($r, $c) = $m->shape();
        $exp = new Matrix($r, $c);
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $exp->row[$i][$j] = exp($m->row[$i][$j]);
            }
        }
        $result = new Matrix($r, $c);
        for ($i = 0; $i < $r; $i++) {
            $sum = array_sum($exp->row[$i]);
            for ($j = 0; $j < $c; $j++) {
                $result->row[$i][$j] = $exp->row[$i][$j] / $sum;
            }
        }
        return $result;
    }

    private static function _cross_entropy_error(Matrix $y, Matrix $t) {
        list($r, $c) = $y->shape();
        $batchSize = $r;
        $t_logy = new Matrix($r, $c);
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $t_logy->row[$i][$j] = $t->row[$i][$j] * log($y->row[$i][$j]);
            }
        }
        $result = 0;
        for ($i = 0; $i < $r; $i++) {
            $result += -1 * array_sum($t_logy->row[$i]);
        }
        $result /= $batchSize;
        return $result;
    }

}

/*
$images = Util::downloadMnist();
Util::loadData();

$index = $_GET['index'];
Util::outputTrainImage($index);
 */

function main() {
    $itersNum = 10000;
    $trainSize = 60000;
    $batchSize = 100;
    $learningRate = 0.1;

    $trainLossList = [];

    Util::loadData();
    $network = new TwoLayerNeuralNet($inputSize=784, $hiddenSize=10, $outputSize=10);

    for ($i = 0; $i < 1; $i++) {
        // ミニバッチの取得
        $batchMask = Util::randoms($trainSize, $batchSize);
        list($xBatch, $tBatch) = Util::getTrainBatch($batchMask);

        echo "numericalGradient: i = $i\n";
        $grad = $network->numericalGradient($xBatch, $tBatch);

        foreach (['W1', 'b1', 'W2', 'b2'] as $key) {
            $network->params[$key] -= $learningRate * $grad[$key];
        }

        $loss = $network->loss($xBatch, $tBatch);

        echo "(i, loss) = ($i, $loss)\n";
        $trainLossList[] = $loss;
    }
}

main();