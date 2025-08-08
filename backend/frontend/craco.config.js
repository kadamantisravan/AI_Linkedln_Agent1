const webpack = require('webpack');

module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      webpackConfig.resolve.fallback = {
        ...webpackConfig.resolve.fallback,
        http: require.resolve("stream-http"),
        https: require.resolve("https-browserify"),
        stream: require.resolve("stream-browserify"),
        zlib: require.resolve("browserify-zlib"),
        util: require.resolve("util"),
        url: require.resolve("url"),
        crypto: require.resolve("crypto-browserify"),
        assert: require.resolve("assert"),
        buffer: require.resolve("buffer/"),
        process: require.resolve("process/browser.js"), // ✅ use `.js` here
      };
      return webpackConfig;
    },
  },
  plugins: [
    {
      plugin: {
        overrideWebpackConfig: ({ webpackConfig }) => {
          webpackConfig.plugins.push(
            new webpack.ProvidePlugin({
              process: 'process/browser.js', // ✅ use `.js` here too
              Buffer: ['buffer', 'Buffer'],
            })
          );
          return webpackConfig;
        },
      },
    },
  ],
};
