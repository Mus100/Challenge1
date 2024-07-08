const express = require('express');
const router = express.Router();
const Character = require('../models/character');
const { characterValidationRules, validate } = require('../middleware/validation');

// Get a single character by ID
router.get('/:id', async (req, res) => {
  try {
    const character = await Character.findById(req.params.id);
    if (!character) {
      return res.status(404).send({ error: 'Character not found' });
    }
    res.send(character);
  } catch (error) {
    res.status(500).send({ error: 'Internal Server Error' });
  }
});

// Get all characters with pagination
router.get('/', async (req, res) => {
  const page = parseInt(req.query.page, 10) || 1;
  const limit = parseInt(req.query.limit, 10) || 10;
  const skip = (page - 1) * limit;

  try {
    const characters = await Character.find().skip(skip).limit(limit);
    res.send(characters);
  } catch (error) {
    res.status(500).send({ error: 'Internal Server Error' });
  }
});

// Create a new character
router.post('/', characterValidationRules(), validate, async (req, res) => {
  try {
    const character = new Character(req.body);
    await character.save();
    res.status(201).send(character);
  } catch (error) {
    res.status(400).send({ error: 'Bad Request' });
  }
});

// Update a character by ID
router.put('/:id', characterValidationRules(), validate, async (req, res) => {
  try {
    const character = await Character.findByIdAndUpdate(req.params.id, req.body, { new: true, runValidators: true });
    if (!character) {
      return res.status(404).send({ error: 'Character not found' });
    }
    res.send(character);
  } catch (error) {
    res.status(400).send({ error: 'Bad Request' });
  }
});

// Delete a character by ID
router.delete('/:id', async (req, res) => {
  try {
    const character = await Character.findByIdAndDelete(req.params.id);
    if (!character) {
      return res.status(404).send({ error: 'Character not found' });
    }
    res.send({ message: 'Character deleted' });
  } catch (error) {
    res.status(500).send({ error: 'Internal Server Error' });
  }
});

module.exports = router;
