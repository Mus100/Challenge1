
const mongoose = require('mongoose');

const characterSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
    },
    role: {
        type: String,
        required: true,
    },
    level: {
        type: Number,
        required: true,
        min: 1,
    }
});

const Character = mongoose.model('Character', characterSchema);

module.exports = Character;
