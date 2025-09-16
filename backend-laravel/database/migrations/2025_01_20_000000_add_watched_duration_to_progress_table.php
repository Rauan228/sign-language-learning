<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('progress', function (Blueprint $table) {
            $table->integer('watched_duration')->default(0)->after('time_spent_minutes'); // время просмотра в секундах
            $table->boolean('is_completed')->default(false)->after('watched_duration'); // флаг завершения урока
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('progress', function (Blueprint $table) {
            $table->dropColumn(['watched_duration', 'is_completed']);
        });
    }
};