// ExpressJS imports
import express from 'express';
// wtf_wikipedia imports
import bodyParser from 'body-parser';
import wtf from 'wtf_wikipedia';
//const wtf = require('wtf_wikipedia');

const app = express();
app.use(bodyParser.json({limit: '999mb', type: '*/*'}));

const port = parseInt(process.env.SERVER_PORT || "3000");


// API proxy
app.post('/toJSON', function(req, res) {
    // init default values
    var result = null;
    var success = false;

    // try parsing
    try {
        result = wtf(req.body.text).json();
        success = true;
    } catch (e) {
        // Indicate failure
        result = e.toString();
        success = false;
    }

    // return results
    res.send({
        "result": result,
        "success": success
    });
  });

// Listen for interceptions
process.on('SIGINT', function() {
  process.exit();
});

// Run event loop
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
