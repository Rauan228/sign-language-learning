<?php

require_once 'vendor/autoload.php';

$app = require_once 'bootstrap/app.php';
$app->make(Illuminate\Contracts\Console\Kernel::class)->bootstrap();

use Illuminate\Support\Facades\DB;

echo "Курсы и их цены:\n";
echo "==================\n";

$courses = DB::table('courses')->select('title', 'price')->get();

foreach ($courses as $course) {
    echo $course->title . ' - ' . number_format($course->price, 2) . ' руб.' . "\n";
}

echo "\nВсего курсов: " . $courses->count() . "\n";