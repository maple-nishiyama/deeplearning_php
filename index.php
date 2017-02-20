<?php

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
        $numImage = $header[2];
        $numRow = $header[3];
        $numCol = $header[4];

        $pixels = $numRow * $numCol;
        $start = 32 * 4 + $pixels * $index;
        return array_values(unpack("C{$pixels}", substr($image, $start, $pixels)));
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
        //l;header("Content-type: image/png");
        $pixels = self::getImageAt(self::$__mnist_train_image_cache, $index);
        self::drawImage($pixels, 28, 28);
    }
}

$images = Util::downloadMnist();
Util::loadData();

$index = $_GET['index'];
Util::outputTrainImage($index);