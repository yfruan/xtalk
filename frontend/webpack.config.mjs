import path from 'path';
import HtmlWebpackPlugin from 'html-webpack-plugin';

export default {
    entry: {
        main: ['./src/index.js', './src/index.ts'],
    },
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'dist')
    },
    resolve: {
        extensions: ['.ts', '.js'],
    },
    module: {
        rules: [
            {
                test: /\.ts$/,
                use: 'ts-loader',
                exclude: /node_modules/,
            },
            {
                test: /\.onnx$/,
                type: 'asset/resource',
                generator: {
                    filename: 'models/[name].[hash][ext]',
                },
            }
        ]
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: './example/index.html',
            filename: 'index.html',
        }),
    ],
    devServer: {
        static: path.resolve('dist'),
        compress: true,
        port: 7634,
    },
};
