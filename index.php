<?php
ini_set('memory_limit', '1G');

define('DS', DIRECTORY_SEPARATOR);

function dumpMatrix(Matrix $m, $filename) {
    $out = '';
    $a = $m->toArray();
    foreach ($a as $row) {
        $out .= implode(', ', array_map(function($e) { return sprintf("%.4f", $e); }, $row)) . "\n";
    }
    file_put_contents($filename, $out);
}
class Util {

    const MNIST_TRAIN_IMAGE_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz';
    const MNIST_TRAIN_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz';
    const MNIST_TEST_IMAGE_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz';
    const MNIST_TEST_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz';

    const MNIST_DATA_DIR = 'mnist';
    const MNIST_TRAIN_IMAGE_FILE = 'mnist_train_image';
    const MNIST_TRAIN_LABEL_FILE = 'mnist_train_label';
    const MNIST_TEST_IMAGE_FILE = 'mnist_test_image';
    const MNIST_TEST_LABEL_FILE = 'mnist_test_label';

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

    static function getImageAt($image, $index, $normalize = true) {
        // ヘッダー解析
        $header = unpack('N4', $image);
        $magic = $header[1];
        $numImages = $header[2];
        $numRow = $header[3];
        $numCol = $header[4];

        $pixels = $numRow * $numCol;
        $start = 4 * 4 + $pixels * $index;
        $result = array_values(unpack("C{$pixels}", substr($image, $start, $pixels)));
        if ($normalize) {
            $result = array_map(function($pixel) { return $pixel / 255.0;}, $result);
        }
        return $result;
    }

    static function getLabelAt($label, $index, $onehot = true) {
        $header = unpack('N2', $label);
        $magic = $header[1];
        $numLabels = $header[2];
        $label = unpack('C', substr($label, $index + 2 * 4, 1));
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
        return Matrix::createFromData($data);
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

    static function outputTrainLabel($index) {
        $lb = self::getLabelAt(self::$__mnist_train_label_cache, $index, $onehot = true);
        return implode(', ', $lb);
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

    /**
     * Matrix の成分をランダム値にする
     */
    public static function randomize(Matrix $m) {
        list($r, $c) = $m->shape();
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $m->set($i, $j, mt_rand(0, 1E4) / 3E6);
            }
        }
        return $m;
    }
}

interface Layer {

    public function forward(Matrix $x);

    public function backward(Matrix $dout);
}

class Relu implements Layer {

    public function __construct() {
        $this->x = null;
    }

    public function forward(Matrix $x) {
        $this->x = $x;
        list($r, $c) = $x->shape();
        $result = new Matrix($r, $c);
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $val = $x->get($i, $j) > 0 ? $x->get($i, $j) : 0;
                $result->set($i, $j, $val);
            }
        }
        return $result;
    }

    public function backward(Matrix $dout) {
        list($r, $c) = $dout->shape();
        $result = new Matrix($r, $c);
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $val = $this->x->get($i, $j) > 0 ? $dout->get($i, $j) : 0;
                $result->set($i, $j, $val);
            }
        }
        return $result;
    }

}

class Sigmoid implements Layer {

    public function __construct() {
        $this->out = null;
    }

    public function forward(Matrix $x, Matrix $t = null) {
        $this->out = self::_sigmoid($x);
        return $this->out;
    }

    public function backward(Matrix $dout) {
        $ones = Matrix::onesLike($this->out);
        $dx = $dout
                ->componentwise_prod($ones->minus($this->out))
                ->componentwise_prod($this->out);
        return $dx;
    }

    private static function _sigmoid(Matrix $m) {
        list($r, $c) = $m->shape();
        $result = new Matrix($r, $c);
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $val = 1 / (1 + exp($m->get($i, $j)));
                $result->set($i, $j, $val);
            }
        }
        return $result;
    }
}

class Affine implements Layer {

    public function __construct(Matrix &$W, Matrix &$b) {
        $this->W =& $W;
        $this->b =& $b;
        $this->x = null;
        $this->dW = null;
        $this->db = null;
    }

    public function forward(Matrix $x) {
        $this->x = $x;
        $out = $x->mul($this->W)->plus($this->b);
        return $out;
    }

    public function backward(Matrix $dout) {
        $dx = $dout->mul($this->W->transpose());
        $this->dW = $this->x->transpose()->mul($dout);
        $this->db = $dout->sumRow();
        return $dx;
    }
}

class SoftmaxWithLoss implements Layer {

    public function __construct() {
        $this->loss = null;
        $this->y = null;
        $this->t = null;
    }

    public function forward(Matrix $x, Matrix $t = null) {
        $this->t = $t;
        $this->y = self::softmax($x);
        $this->loss = self::cross_entropy_error($this->y, $this->t);
        return $this->loss;
    }

    public function backward(Matrix $dout) {
        $batchSize = $this->t->shape()[0];
        $dx = $this->y->minus($this->t)->scale(1/$batchSize);
        return $dx;
    }


    public static function softmax(Matrix $m) {
        list($r, $c) = $m->shape();
        $exp = new Matrix($r, $c);
        // 最大値を引き去るために、先に求めておく
        $max = 0;
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $max = max($max, $m->get($i, $j));
            }
        }
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $exp->set($i, $j, exp($m->get($i, $j) - $max));
            }
        }
        $result = new Matrix($r, $c);
        for ($i = 0; $i < $r; $i++) {
            $sum = array_sum($exp->toArray()[$i]);
            for ($j = 0; $j < $c; $j++) {
                $result->set($i, $j, $exp->get($i, $j) / $sum);
            }
        }
        return $result;
    }

    public static function cross_entropy_error(Matrix $y, Matrix $t) {
        list($r, $c) = $y->shape();
        $batchSize = $r;
        $t_logy = new Matrix($r, $c);
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $val = $t->get($i, $j) * log($y->get($i, $j));
                $t_logy->set($i, $j, $val);
            }
        }
        $result = 0;
        for ($i = 0; $i < $r; $i++) {
            $result += -1 * array_sum($t_logy->toArray()[$i]);
        }
        $result /= $batchSize;
        return $result;
    }
}


class TwoLayerNeuralNet {

    public $params = [];

    public $layers = [];

    public function __construct($inputSize, $hiddenSize, $outputSize, $weightInitStd=0.01) {
        $this->params = [];
        $this->params['W1'] = Util::randomize(new Matrix($inputSize, $hiddenSize));
        $this->params['b1'] = new Matrix(1, $hiddenSize);
        $this->params['W2'] = Util::randomize(new Matrix($hiddenSize, $outputSize));
        $this->params['b2'] = new Matrix(1, $outputSize);

        // レイヤーの生成
        $this->layers = [];
        $this->layers['Affine1'] = new Affine($this->params['W1'], $this->params['b1']);
        $this->layers['Relu1'] = new Relu();
        $this->layers['Affine2'] = new Affine($this->params['W2'], $this->params['b2']);
        $this->lastLayer = new SoftmaxWithLoss();
    }

    public function predict(Matrix $x) {
        foreach($this->layers as $key => $layer) {
            $x = $layer->forward($x);
        }
        return $x;
    }

    public function loss(Matrix $x, Matrix $t) {
        $y = $this->predict($x);
        return $this->lastLayer->forward($y, $t);
    }

    public function batchAccuracy(Matrix $x, Matrix $t) {
        $y = $this->predict($x);
        $ymax = $y->argmax($dir=1);
        $tmax = $t->argmax($dir=1);
        $r = $y->shape()[0];
        $acc = 0;
        for ($i = 0; $i < $r; $i++) {
            if ($ymax->get($i, 0) === $tmax->get($i, 0)) {
                $acc += 1;
            }
        }
        return $acc;
    }

    public function trainAccuracy() {
        $acc = 0;
        $batchSize = 100;
        $totalSize = 60000;
        for ($i = 0; $i < $totalSize; $i += $batchSize) {
            $mask = range($i, min($i + $batchSize, $totalSize - 1));
            list($xTrain, $tTrain) = Util::getTrainBatch($mask);
            $acc += $this->batchAccuracy($xTrain, $tTrain);
        }
        return $acc / $totalSize;
    }

    public function testAccuracy() {
        $acc = 0;
        $batchSize = 100;
        $totalSize = 10000;
        for ($i = 0; $i < $totalSize; $i += $batchSize) {
            $mask = range($i, min($i + $batchSize, $totalSize - 1));
            list($xTest, $tTest) = Util::getTestBatch($mask);
            $acc += $this->batchAccuracy($xTest, $tTest);
        }
        return $acc / $totalSize;
    }

    // 数値微分
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
                $tmpVal = $x->get($i, $j);

                $x->set($i, $j, $tmpVal + $h);
                $fxh1 = $f($x);

                $x->set($i, $j, $tmpVal - $h);
                $fxh2 = $f($x);

                $grad->set($i, $j, ($fxh1 - $fxh2) / (2 * $h));
                $x->set($i, $j, $tmpVal);
            }
        }

        return $grad;
    }

    // 誤差逆伝搬法
    public function backpropagationGradient(Matrix $x, Matrix $t) {

        // まず一回順伝搬させる
        $this->loss($x, $t);

        $dout = new Matrix(1, 1);
        $dout->set(0, 0, 1);
        $dout = $this->lastLayer->backward($dout);

        $reversedLayers = array_reverse($this->layers);
        foreach ($reversedLayers as $key => $layer) {
            $dout = $layer->backward($dout);
        }

        $grads = [];
        $grads['W1'] = $this->layers['Affine1']->dW;
        $grads['b1'] = $this->layers['Affine1']->db;
        $grads['W2'] = $this->layers['Affine2']->dW;
        $grads['b2'] = $this->layers['Affine2']->db;

        return $grads;
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
    $trainAccList = [];
    $testAccList = [];

    $iterPerEpoch = max($trainSize / $batchSize, 1);

    Util::loadData();
    $network = new TwoLayerNeuralNet($inputSize=784, $hiddenSize=50, $outputSize=10);

    for ($i = 0; $i < $itersNum; $i++) {
        // ミニバッチの取得
        $batchMask = Util::randoms($trainSize, $batchSize);
        list($xBatch, $tBatch) = Util::getTrainBatch($batchMask);

        //$grad = $network->numericalGradient($xBatch, $tBatch);
        $grad = $network->backpropagationGradient($xBatch, $tBatch);

        foreach (['W1', 'b1', 'W2', 'b2'] as $key) {
            $network->params[$key] = $network->params[$key]->plus($grad[$key]->scale(-$learningRate));
        }

        $loss = $network->loss($xBatch, $tBatch);

        $trainLossList[] = $loss;

        if ($i % $iterPerEpoch == 0) {
            $trainAcc = $network->trainAccuracy();
            $testAcc = $network->testAccuracy();
            $trainAccList[] = $trainAcc;
            $testAccList[] = $testAccList;
            echo "(trainAcc, testAcc) = ($trainAcc, $testAcc)\n";
        }
    }
}
main();
