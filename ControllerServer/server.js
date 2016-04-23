var net = require('net');
var rl = require('readline');

var server = net.createServer(function(socket) {
	function emitLines (stream) {
	  var backlog = ''
	  stream.on('data', function (data) {
	    backlog += data
	    console.log(data);
	    var n = backlog.indexOf('\n')
	    // got a \n? emit one or more 'line' events
	    while (~n) {
	      stream.emit('line', backlog.substring(0, n))
	      backlog = backlog.substring(n + 1)
	      n = backlog.indexOf('\n')
	    }
	  })
	  stream.on('end', function () {
	    if (backlog) {
	      stream.emit('line', backlog)
	    }
	  })
	}
	emitLines(socket);
	socket.on("line", function(line){
		console.log(line);
	})
});

server.listen(8080, '10.0.0.7');