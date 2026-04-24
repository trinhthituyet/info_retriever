# UI/UX Redesign Prompt

You are a senior UI/UX designer specializing in modern SaaS, document-management, and AI-powered products.

I will provide a screenshot of a webpage I have already implemented. Your task is to critique the existing UI and redesign it into a significantly better version, while preserving the core product requirements and functionality described below.

## Product Context

The product is a personal document intelligence and search application.

Users have a folder of personal documents in mixed formats, including:

- PDFs, including scanned PDFs
- DOCX files
- Photos/images
- Plain-text files

Typical documents include:

- Rental contracts
- Employment contracts
- Insurance policies

Users should be able to upload these documents into a searchable store, then ask questions about them in natural language.

## Core Capabilities

### 1. Add Documents

Users can upload one or multiple documents.

The system should:

- Accept multiple file formats
- Extract text, including OCR for scanned documents
- Extract relevant structured fields and metadata
- Index the content for search and retrieval
- Clearly communicate upload, processing, and indexing status
- Handle errors and unsupported files gracefully

### 2. Ask Questions

Users can have a conversational interaction with their documents rather than asking isolated questions.

The conversation must maintain context across follow-up questions.

For example:

> User: What is the monthly rent?
>
> AI: The monthly rent is $2,500.
>
> User: And the deposit?
>
> AI: The security deposit is $5,000.

Answers should include clear citations showing:

- Document name
- Page number
- Relevant source wording

## Your Design Task

Analyze the provided screenshot carefully and identify problems with:

- Information hierarchy
- Layout and spacing
- Navigation
- Visual hierarchy
- Typography
- Color and contrast
- Component design
- Upload/document management UX
- AI conversation UX
- Citation/source presentation
- Empty, loading, and error states
- Discoverability of the two core capabilities
- Overall usability and clarity
- Modernity and visual polish

Then redesign the page to address these issues.

## Design Goals

The redesigned UI should feel:

- Modern
- Clean
- Premium
- Trustworthy
- Intuitive
- Calm and focused
- Appropriate for handling sensitive personal documents
- Like a polished production SaaS application

Prioritize clarity and usability over visual decoration.

The two primary workflows — adding documents and asking questions — should be immediately obvious.

The AI answer experience should make citations feel trustworthy and easy to verify without overwhelming the conversation.

## Important Constraints

- Preserve the core product concept and functionality.
- Do not introduce unrelated features simply for visual appeal.
- Improve the existing design rather than creating an entirely unrelated product.
- Use realistic UI content rather than generic placeholder text where possible.
- Ensure the redesigned interface looks like a real, production-ready application.
- Pay particular attention to responsive layout, spacing, alignment, accessibility, and interaction affordances.

## Output

First, internally analyze the screenshot and determine what should be improved.

Then produce the redesigned webpage as an image.

The output should be a high-fidelity UI mockup, not a written explanation or wireframe.

## Existing Screenshot

Here is the screenshot of the webpage I currently have implemented:
