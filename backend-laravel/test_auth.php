<?php

require_once 'vendor/autoload.php';

$app = require_once 'bootstrap/app.php';
$app->make(Illuminate\Contracts\Console\Kernel::class)->bootstrap();

$user = App\Models\User::first();
if ($user) {
    $token = $user->createToken('test')->plainTextToken;
    echo "User: " . $user->email . "\n";
    echo "Token: " . $token . "\n";
} else {
    echo "No users found\n";
}