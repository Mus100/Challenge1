
const { body, validationResult } = require('express-validator');

const characterValidationRules = () => {
    return [
        body('name').isString().withMessage('Name must be a string'),
        body('role').isString().withMessage('Role must be a string'),
        body('level').isInt({ min: 1 }).withMessage('Level must be an integer greater than 0')
    ];
};

const validate = (req, res, next) => {
    const errors = validationResult(req);
    if (errors.isEmpty()) {
        return next();
    }
    const extractedErrors = [];
    errors.array().map(err => extractedErrors.push({ [err.param]: err.msg }));

    return res.status(422).json({
        errors: extractedErrors,
    });
};

module.exports = {
    characterValidationRules,
    validate,
};
