
const notFoundHandler = (req, res, next) => {
  res.status(404).send({ error: 'Not Found' });
};

const errorHandler = (err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send({ error: 'Internal Server Error' });
};

module.exports = {
  notFoundHandler,
  errorHandler,
};
