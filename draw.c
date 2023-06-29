#import "draw.h"

void vizualize_picture(MNISTData *data){
    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = IMAGE_SIZE * SCALE_FACTOR;
    const int screenHeight = IMAGE_SIZE * SCALE_FACTOR + 200;

    InitWindow(screenWidth, screenHeight, "MNIST Image Viewer");

    SetTargetFPS(1);               
    //--------------------------------------------------------------------------------------

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        if (IsKeyPressed(KEY_ESCAPE))
            break;
        BeginDrawing();

            ClearBackground(RAYWHITE);

            // Draw the MNIST image
            for (int x= 0; x < IMAGE_SIZE; ++x) {
                for (int y = 0; y < IMAGE_SIZE; ++y) {
                    // MNIST pixel values are 0-255, so we need to convert this to a color
                    int val = data->image[y][x];
                    Color color = { val, val, val, 255 };
                    DrawRectangle(x * SCALE_FACTOR, y * SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR, color);
                }
            }
        const char* text = "Text at the bottom";
        int textWidth = MeasureText(text, 20);
        int xPos = (screenWidth - textWidth) / 2;
        int yPos = screenHeight - 50;  // Adjust the vertical position as needed

        DrawText(text, xPos, yPos, 20, BLACK);

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------   
    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------
}



