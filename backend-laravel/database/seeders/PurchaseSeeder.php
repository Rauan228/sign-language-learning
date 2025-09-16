<?php

namespace Database\Seeders;

use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use App\Models\Purchase;
use App\Models\User;
use App\Models\Course;
use Carbon\Carbon;

class PurchaseSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Получаем пользователей и курсы
        $users = User::all();
        $courses = Course::all();
        
        if ($users->isEmpty() || $courses->isEmpty()) {
            $this->command->info('No users or courses found. Please run UserSeeder and CourseSeeder first.');
            return;
        }
        
        // Создаем покупки для каждого пользователя
        foreach ($users as $user) {
            // Каждый пользователь покупает 1-3 случайных курса
            $coursesToPurchase = $courses->random(rand(1, min(3, $courses->count())));
            
            foreach ($coursesToPurchase as $course) {
                Purchase::create([
                    'user_id' => $user->id,
                    'course_id' => $course->id,
                    'amount' => $course->price,
                    'status' => 'completed',
                    'payment_method' => 'card',
                    'transaction_id' => 'txn_' . uniqid(),
                    'purchased_at' => Carbon::now()->subDays(rand(1, 30)),
                    'created_at' => Carbon::now(),
                    'updated_at' => Carbon::now(),
                ]);
            }
        }
        
        $this->command->info('Purchase seeder completed successfully!');
    }
}
